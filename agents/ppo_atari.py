"""CleanRL-style PPO for Atari, adapted for JEPA encoder-swap experiments.

Supports three encoder modes:
  - Stock CNN (default): standard Nature-CNN trained end-to-end.
  - Frozen encoder: pretrained encoder (JEPA/AE) is frozen, only heads train.
  - Fine-tune encoder: pretrained encoder trains end-to-end with a lower LR.

Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
"""

import time
import random
from pathlib import Path

import ale_py  # noqa: F401  — registers ALE environments
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Default config (CleanRL Atari defaults)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "env_id": "ALE/Breakout-v5",
    "total_timesteps": 10_000_000,
    "learning_rate": 2.5e-4,
    "num_envs": 8,
    "num_steps": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 4,
    "update_epochs": 4,
    "clip_coef": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "norm_adv": True,
    "seed": 1,
    "capture_video": False,
    "run_name": None,
    "save_dir": "results/v0",
    "save_interval": 500_000,
    "freeze_encoder": True,
    "encoder_lr_scale": 0.1,
}


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_id, seed, idx, capture_video=False, run_name=""):
    """Return a thunk that creates a single Atari env with standard wrappers."""

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda t: t % 100 == 0
            )
        else:
            env = gym.make(env_id, frameskip=1)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization (CleanRL convention)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariEncoder(nn.Module):
    """Standard Nature-CNN encoder for 84x84 Atari frames.

    Input:  (batch, 4, 84, 84) float32 in [0, 1]
    Output: (batch, 512) feature vector

    This module is deliberately separate so it can be swapped for a JEPA
    or autoencoder in later experiment phases.
    """

    FEATURE_DIM = 512

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class Agent(nn.Module):
    """PPO agent: encoder + actor head + critic head."""

    def __init__(self, num_actions, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else AtariEncoder()
        feat_dim = self.encoder.FEATURE_DIM
        frozen = encoder is not None

        if frozen:
            # MLP heads for frozen encoder: features may not be linearly
            # separable for action/value prediction.
            self.actor = nn.Sequential(
                layer_init(nn.Linear(feat_dim, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, num_actions), std=0.01),
            )
            self.critic = nn.Sequential(
                layer_init(nn.Linear(feat_dim, 256)),
                nn.ReLU(),
                layer_init(nn.Linear(256, 1), std=1.0),
            )
        else:
            self.actor = layer_init(nn.Linear(feat_dim, num_actions), std=0.01)
            self.critic = layer_init(nn.Linear(feat_dim, 1), std=1.0)

    @staticmethod
    def preprocess(obs):
        """Convert observations from uint8 NHWC to float32 NCHW in [0, 1]."""
        # obs: (batch, 84, 84, 4) uint8  ->  (batch, 4, 84, 84) float32
        x = torch.as_tensor(obs, dtype=torch.float32)
        if x.ndim == 4 and x.shape[-1] in (1, 4):
            x = x.permute(0, 3, 1, 2)
        return x / 255.0

    def encode(self, obs):
        """Preprocess obs and run through encoder. Returns feature vector."""
        x = self.preprocess(obs)
        return self.encoder(x)

    def get_value(self, obs):
        features = self.encode(obs)
        return self.critic(features)

    def get_action_and_value(self, obs, action=None):
        features = self.encode(obs)
        return self.action_and_value_from_features(features, action)

    def get_value_from_features(self, features):
        """Critic forward from pre-computed features (skips encoder)."""
        return self.critic(features)

    def action_and_value_from_features(self, features, action=None):
        """Actor+critic forward from pre-computed features (skips encoder)."""
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------

def train(config=None, encoder=None):
    """Run PPO training. Returns the path to the final saved checkpoint.

    Args:
        config: Training configuration dict. Missing keys filled from DEFAULT_CONFIG.
            Set freeze_encoder=False and encoder_lr_scale to fine-tune an encoder.
        encoder: Optional pre-trained encoder (e.g. VisionTransformer). Behavior
            depends on config["freeze_encoder"] (default True = frozen, heads only).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Derived values
    batch_size = cfg["num_envs"] * cfg["num_steps"]
    minibatch_size = batch_size // cfg["num_minibatches"]
    num_updates = cfg["total_timesteps"] // batch_size

    # Seeding
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = get_device()
    run_name = cfg["run_name"] or f"ppo_{cfg['env_id'].split('/')[-1]}_{seed}_{int(time.time())}"
    save_dir = Path(cfg["save_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(str(save_dir / "tb"))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in cfg.items()),
    )

    # Vectorized environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg["env_id"], seed, i, cfg["capture_video"], run_name)
         for i in range(cfg["num_envs"])]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    num_actions = envs.single_action_space.n

    # Agent and optimizer
    freeze_encoder = encoder is not None and cfg["freeze_encoder"]
    if encoder is not None:
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()
        agent = Agent(num_actions, encoder=encoder).to(device)
        head_params = list(agent.actor.parameters()) + list(agent.critic.parameters())
        if freeze_encoder:
            optimizer = optim.Adam(head_params, lr=cfg["learning_rate"], eps=1e-5)
        else:
            # Fine-tune: lower LR on encoder to avoid catastrophic forgetting
            optimizer = optim.Adam([
                {"params": list(agent.encoder.parameters()),
                 "lr": cfg["learning_rate"] * cfg["encoder_lr_scale"]},
                {"params": head_params, "lr": cfg["learning_rate"]},
            ], eps=1e-5)
        num_encoder = sum(p.numel() for p in encoder.parameters())
        num_heads = sum(p.numel() for p in head_params)
        mode = "frozen" if freeze_encoder else f"fine-tune (encoder_lr_scale={cfg['encoder_lr_scale']})"
        print(f"Encoder: {num_encoder:,} params ({mode}) | Heads: {num_heads:,} params")
        # Save encoder architecture info so _load_checkpoint can reconstruct it
        cfg["encoder_type"] = "vit"
        cfg["encoder_in_channels"] = encoder.patch_embed.proj.in_channels
        cfg["encoder_patch_size"] = encoder.patch_embed.patch_size
        cfg["encoder_embed_dim"] = encoder.embed_dim
        cfg["encoder_num_heads"] = encoder.blocks[0].attn.num_heads
        cfg["encoder_num_layers"] = len(encoder.blocks)
        cfg["encoder_feature_dim"] = encoder.feature_dim
    else:
        agent = Agent(num_actions).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    # Rollout storage
    obs_buf = torch.zeros(
        (cfg["num_steps"], cfg["num_envs"]) + envs.single_observation_space.shape,
        dtype=torch.uint8,
    )
    actions_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]), dtype=torch.long)
    logprobs_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    rewards_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    dones_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    values_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    # Cache encoder features when encoder is frozen (avoids redundant forward
    # passes during PPO update — 17x speedup for ViT on MPS).
    feat_dim = agent.encoder.FEATURE_DIM
    features_buf = torch.zeros(
        (cfg["num_steps"], cfg["num_envs"], feat_dim)
    ) if freeze_encoder else None

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs)
    next_done = torch.zeros(cfg["num_envs"])
    initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

    for update in range(1, num_updates + 1):
        # ---- Learning rate annealing (linear) ----
        frac = 1.0 - (update - 1.0) / num_updates
        for pg, init_lr in zip(optimizer.param_groups, initial_lrs):
            pg["lr"] = frac * init_lr

        # ---- Rollout phase ----
        for step in range(cfg["num_steps"]):
            global_step += cfg["num_envs"]
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                if freeze_encoder:
                    feats = agent.encode(next_obs.to(device))
                    features_buf[step] = feats.cpu()
                    action, logprob, _, value = agent.action_and_value_from_features(feats)
                else:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs.to(device)
                    )
                values_buf[step] = value.flatten().cpu()

            actions_buf[step] = action.cpu()
            logprobs_buf[step] = logprob.cpu()

            next_obs, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            rewards_buf[step] = torch.as_tensor(reward)
            next_obs = torch.as_tensor(next_obs)
            next_done = torch.as_tensor(terminated | truncated, dtype=torch.float32)

            # Log completed episodes
            # gymnasium >= 1.0: episode stats are in infos["episode"] with
            # a boolean mask infos["_episode"] indicating which envs finished.
            # Older gymnasium used infos["final_info"] instead.
            if "_episode" in infos:
                finished = infos["_episode"]
                for i, done_flag in enumerate(finished):
                    if done_flag:
                        ep_r = float(infos["episode"]["r"][i])
                        ep_l = int(infos["episode"]["l"][i])
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={ep_r:.1f}, "
                            f"episodic_length={ep_l}"
                        )
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)
                        writer.add_scalar("charts/episodic_length", ep_l, global_step)
            elif "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        ep_r = info["episode"]["r"]
                        ep_l = info["episode"]["l"]
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={ep_r:.1f}, "
                            f"episodic_length={ep_l}"
                        )
                        writer.add_scalar("charts/episodic_return", ep_r, global_step)

        # ---- GAE advantage computation ----
        with torch.no_grad():
            if freeze_encoder:
                next_feats = agent.encode(next_obs.to(device))
                next_value = agent.get_value_from_features(next_feats).flatten().cpu()
            else:
                next_value = agent.get_value(next_obs.to(device)).flatten().cpu()
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(cfg["num_steps"])):
                if t == cfg["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = (
                    rewards_buf[t]
                    + cfg["gamma"] * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # ---- Flatten the batch ----
        b_obs = obs_buf.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_features = features_buf.reshape(-1, feat_dim) if freeze_encoder else None

        # ---- PPO update ----
        b_inds = np.arange(batch_size)
        clipfracs = []

        for _epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                if freeze_encoder:
                    _, newlogprob, entropy, newvalue = agent.action_and_value_from_features(
                        b_features[mb_inds].to(device),
                        b_actions[mb_inds].to(device),
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds].to(device),
                        b_actions[mb_inds].to(device),
                    )
                logratio = newlogprob - b_logprobs[mb_inds].to(device)
                ratio = logratio.exp()

                with torch.no_grad():
                    # Approximate KL for logging
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds].to(device)
                if cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss (clipped surrogate)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"]
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds].to(device)) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - cfg["ent_coef"] * entropy_loss + cfg["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                if not freeze_encoder and encoder is not None:
                    # Track pre-clip encoder grad norm (last minibatch value logged below)
                    _enc_grad_norm = nn.utils.clip_grad_norm_(
                        agent.encoder.parameters(), float("inf")
                    )
                nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                optimizer.step()

        # ---- Logging ----
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[-1]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if not freeze_encoder and encoder is not None:
            writer.add_scalar("encoder/grad_norm", _enc_grad_norm.item(), global_step)
            writer.add_scalar("charts/encoder_lr", optimizer.param_groups[0]["lr"], global_step)

        if update % 10 == 0:
            print(
                f"update {update}/{num_updates} | "
                f"step {global_step}/{cfg['total_timesteps']} | "
                f"SPS {sps} | "
                f"pg_loss {pg_loss.item():.4f} | "
                f"v_loss {v_loss.item():.4f} | "
                f"entropy {entropy_loss.item():.4f}"
            )

        # ---- Periodic checkpoint ----
        if cfg["save_interval"] > 0 and global_step % cfg["save_interval"] < batch_size:
            _save_checkpoint(agent, optimizer, global_step, cfg, save_dir / f"checkpoint_{global_step}.pt")

    # ---- Final save ----
    final_path = save_dir / "final_model.pt"
    _save_checkpoint(agent, optimizer, global_step, cfg, final_path)
    print(f"Training complete. Final model saved to {final_path}")

    envs.close()
    writer.close()
    return str(final_path)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(agent, optimizer, global_step, config, path):
    """Save model + optimizer state, step count, and config."""
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "config": config,
        },
        path,
    )
    print(f"  [checkpoint] saved to {path}")


def _load_checkpoint(path, device="cpu"):
    """Load a checkpoint and reconstruct the Agent.

    Auto-detects encoder type from saved config: if ``encoder_type`` is
    ``"vit"`` the agent is rebuilt with a VisionTransformer encoder,
    otherwise the default AtariEncoder is used.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    sd = ckpt["model_state_dict"]

    # Infer num_actions from actor output weight (handles both Linear and Sequential)
    if "actor.weight" in sd:
        num_actions = sd["actor.weight"].shape[0]
    else:
        # MLP head: actor is Sequential(Linear, ReLU, Linear) — last linear is index 2
        num_actions = sd["actor.2.weight"].shape[0]

    encoder_type = cfg.get("encoder_type", "cnn")
    if encoder_type == "vit":
        from agents.encoder import VisionTransformer
        encoder = VisionTransformer(
            in_channels=cfg.get("encoder_in_channels", 4),
            patch_size=cfg.get("encoder_patch_size", 12),
            embed_dim=cfg.get("encoder_embed_dim", 192),
            num_heads=cfg.get("encoder_num_heads", 3),
            num_layers=cfg.get("encoder_num_layers", 4),
            feature_dim=cfg.get("encoder_feature_dim", 512),
        )
        agent = Agent(num_actions, encoder=encoder).to(device)
    else:
        agent = Agent(num_actions).to(device)

    agent.load_state_dict(ckpt["model_state_dict"])
    return agent, ckpt


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path, env_fn=None, num_episodes=50, device=None):
    """Load a trained model and evaluate it.

    Args:
        model_path: Path to a saved checkpoint (.pt file).
        env_fn: Optional callable that returns a single gym.Env.
                If None, creates the default env from the checkpoint config.
        num_episodes: Number of evaluation episodes.
        device: Torch device. Auto-detected if None.

    Returns:
        dict with keys: mean_reward, std_reward, rewards (list of floats).
    """
    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    agent, ckpt = _load_checkpoint(model_path, device)
    agent.eval()
    cfg = ckpt["config"]

    if env_fn is None:
        env_fn = make_env(cfg["env_id"], seed=cfg["seed"], idx=0)

    rewards = []
    for ep in range(num_episodes):
        env = env_fn()
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
            obs, reward, terminated, truncated, info = env.step(action.item())
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        env.close()

    rewards_arr = np.array(rewards)
    result = {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "rewards": rewards_arr.tolist(),
    }
    print(
        f"Evaluation ({num_episodes} episodes): "
        f"mean={result['mean_reward']:.2f}, std={result['std_reward']:.2f}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
