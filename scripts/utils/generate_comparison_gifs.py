"""Generate side-by-side comparison GIFs: JEPA vs Stock CNN vs Autoencoder.

Creates:
  1. comparison_clean.gif    — 3 models playing clean Breakout side-by-side
  2. comparison_hard.gif     — 3 models under hard visual perturbations
  3. robustness_strip.gif    — JEPA vs CNN, clean on top row, hard on bottom row

Usage:
    uv run python scripts/generate_comparison_gifs.py
    uv run python scripts/generate_comparison_gifs.py --device mps --max-steps 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from agents.ppo_atari import _load_checkpoint
from env.perturbations import make_perturbed_atari_env
from env.wrappers import make_atari_env


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "Stock CNN": "results/v0/ppo_Breakout-v5_42_1770766846/final_model.pt",
    "JEPA": "results/v0/phase3/jepa/jepa_seed1/final_model.pt",
    "Autoencoder": "results/v0/phase3/autoencoder/autoencoder_seed1/final_model.pt",
}

LABEL_COLORS = {
    "Stock CNN": (59, 130, 246),    # blue
    "JEPA": (16, 185, 129),         # green
    "Autoencoder": (239, 68, 68),   # red
}

HEADER_H = 36
PADDING = 4
BG_COLOR = (15, 15, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_agent(model_path: str, device: torch.device):
    agent, _ = _load_checkpoint(model_path, device)
    agent.eval()
    return agent


def get_font(size: int = 16):
    """Try to load a nice monospace font, fall back to default."""
    font_paths = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def play_episode(agent, env, device, max_steps=3000):
    """Play one episode, return list of (rgb_frame, cumulative_reward)."""
    obs, _ = env.reset()
    frames = []
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        rgb = env.render()
        frames.append((rgb, total_reward))

        obs_t = torch.as_tensor(obs, dtype=torch.uint8).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_t)

        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += float(reward)
        done = terminated or truncated
        step += 1

    # Append final frame
    rgb = env.render()
    frames.append((rgb, total_reward))
    env.close()
    return frames


def compose_side_by_side(
    all_frames: dict[str, list[tuple[np.ndarray, float]]],
    font,
    max_frames: int = 600,
) -> list[Image.Image]:
    """Compose frames from multiple models into side-by-side PIL images.

    all_frames: {model_name: [(rgb_array, cumulative_reward), ...]}
    """
    names = list(all_frames.keys())
    # Determine the minimum episode length (truncate to shortest)
    min_len = min(len(all_frames[n]) for n in names)
    min_len = min(min_len, max_frames)

    # Get frame dimensions from first model's first frame
    sample = all_frames[names[0]][0][0]
    fh, fw = sample.shape[:2]

    # Canvas dimensions
    panel_w = fw + PADDING
    canvas_w = panel_w * len(names) + PADDING
    canvas_h = HEADER_H + fh + PADDING * 2

    composed = []
    for i in range(min_len):
        img = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
        draw = ImageDraw.Draw(img)

        for j, name in enumerate(names):
            rgb, score = all_frames[name][i]
            x = PADDING + j * panel_w
            y = HEADER_H + PADDING

            # Draw the game frame
            frame_img = Image.fromarray(rgb)
            img.paste(frame_img, (x, y))

            # Draw header label with colored text
            color = LABEL_COLORS.get(name, (255, 255, 255))
            label = f"{name}: {score:.0f}"
            draw.text((x + 4, 8), label, fill=color, font=font)

        composed.append(img)

    return composed


def compose_grid(
    top_frames: dict[str, list[tuple[np.ndarray, float]]],
    bot_frames: dict[str, list[tuple[np.ndarray, float]]],
    top_label: str,
    bot_label: str,
    font,
    max_frames: int = 600,
) -> list[Image.Image]:
    """2-row grid: top row = one condition, bottom row = another condition."""
    names = list(top_frames.keys())
    min_len = min(
        min(len(top_frames[n]) for n in names),
        min(len(bot_frames[n]) for n in names),
        max_frames,
    )

    sample = top_frames[names[0]][0][0]
    fh, fw = sample.shape[:2]

    panel_w = fw + PADDING
    row_label_w = 70
    canvas_w = row_label_w + panel_w * len(names) + PADDING
    row_h = HEADER_H + fh + PADDING
    canvas_h = row_h * 2 + PADDING

    small_font = get_font(12)

    composed = []
    for i in range(min_len):
        img = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
        draw = ImageDraw.Draw(img)

        for row_idx, (frames_dict, row_name) in enumerate(
            [(top_frames, top_label), (bot_frames, bot_label)]
        ):
            y_off = row_idx * row_h

            # Row label (rotated text is hard, just draw horizontal)
            draw.text(
                (4, y_off + HEADER_H + fh // 2 - 6),
                row_name,
                fill=(180, 180, 180),
                font=small_font,
            )

            for j, name in enumerate(names):
                rgb, score = frames_dict[name][i]
                x = row_label_w + j * panel_w
                y = y_off + HEADER_H + PADDING

                frame_img = Image.fromarray(rgb)
                img.paste(frame_img, (x, y))

                color = LABEL_COLORS.get(name, (255, 255, 255))
                label = f"{name}: {score:.0f}"
                draw.text((x + 4, y_off + 6), label, fill=color, font=font)

        composed.append(img)

    return composed


def save_gif(frames: list[Image.Image], path: Path, fps: int = 30):
    """Save PIL frames as an optimized GIF."""
    # Subsample if too many frames (keep GIF size reasonable)
    skip = max(1, len(frames) // 600)
    frames = frames[::skip]

    duration = max(1000 // fps, 17)  # ms per frame
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  Saved {path} ({len(frames)} frames, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main generation routines
# ---------------------------------------------------------------------------

def best_of_n_episodes(
    agent,
    env_id: str,
    perturbation: str,
    device: torch.device,
    max_steps: int,
    n_tries: int = 8,
    base_seed: int = 42,
) -> list[tuple[np.ndarray, float]]:
    """Play n_tries episodes and return the one with the highest score."""
    best_frames = None
    best_score = -float("inf")

    for i in range(n_tries):
        seed = base_seed + i * 7
        if perturbation == "clean":
            env = make_atari_env(
                env_id, render_mode="rgb_array", training=False, seed=seed,
            )
        else:
            env = make_perturbed_atari_env(
                env_id, perturbation_level=perturbation,
                render_mode="rgb_array", seed=seed,
            )
        frames = play_episode(agent, env, device, max_steps=max_steps)
        score = frames[-1][1]
        if score > best_score:
            best_score = score
            best_frames = frames

    return best_frames


def generate_comparison_gif(
    perturbation: str,
    output_path: Path,
    device: torch.device,
    max_steps: int,
    seed: int = 42,
    n_tries: int = 8,
):
    """Generate a side-by-side comparison GIF for a given perturbation level."""
    print(f"\n--- Generating {perturbation} comparison (best of {n_tries}) ---")
    font = get_font(14)
    all_frames: dict[str, list[tuple[np.ndarray, float]]] = {}

    for name, model_path in MODELS.items():
        print(f"  Playing {name} ({perturbation})...")
        agent = load_agent(model_path, device)
        frames = best_of_n_episodes(
            agent, "ALE/Breakout-v5", perturbation, device,
            max_steps, n_tries=n_tries, base_seed=seed,
        )
        all_frames[name] = frames
        print(f"    -> {len(frames)} frames, final score: {frames[-1][1]:.0f}")

    composed = compose_side_by_side(all_frames, font)
    save_gif(composed, output_path)


def generate_robustness_grid(
    output_path: Path,
    device: torch.device,
    max_steps: int,
    seed: int = 42,
):
    """Generate a 2x2 grid: (JEPA, CNN) x (clean, hard)."""
    print("\n--- Generating robustness grid ---")
    font = get_font(14)

    subset = {k: v for k, v in MODELS.items() if k in ("JEPA", "Stock CNN")}

    clean_frames: dict[str, list[tuple[np.ndarray, float]]] = {}
    hard_frames: dict[str, list[tuple[np.ndarray, float]]] = {}

    for name, model_path in subset.items():
        agent = load_agent(model_path, device)

        for pert, store in [("clean", clean_frames), ("hard", hard_frames)]:
            print(f"  Playing {name} ({pert}, best of 8)...")
            frames = best_of_n_episodes(
                agent, "ALE/Breakout-v5", pert, device,
                max_steps, n_tries=8, base_seed=seed,
            )
            store[name] = frames
            print(f"    -> {len(frames)} frames, final score: {frames[-1][1]:.0f}")

    composed = compose_grid(
        clean_frames, hard_frames,
        top_label="Clean", bot_label="Hard",
        font=font,
    )
    save_gif(composed, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate comparison GIFs")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--output-dir", default="results/v0/gifs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--fps", type=int, default=30, help="GIF frame rate")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # 1. Clean comparison (all 3 models)
    generate_comparison_gif(
        "clean",
        output_dir / "comparison_clean.gif",
        device, args.max_steps, args.seed,
    )

    # 2. Hard perturbation comparison (all 3 models)
    generate_comparison_gif(
        "hard",
        output_dir / "comparison_hard.gif",
        device, args.max_steps, args.seed,
    )

    # 3. Robustness grid (JEPA vs CNN, clean vs hard)
    generate_robustness_grid(
        output_dir / "robustness_grid.gif",
        device, args.max_steps, args.seed,
    )

    print(f"\nAll GIFs saved to {output_dir}/")


if __name__ == "__main__":
    main()
