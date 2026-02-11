"""Standard Atari preprocessing wrappers for Gymnasium.

Implements the canonical preprocessing pipeline from Mnih et al. (2015) with
minor updates for the Gymnasium API (v0.26+).  Every wrapper is a
``gymnasium.Wrapper`` subclass so they compose cleanly.

Wrapper application order (inside ``make_atari_env``):
    raw env -> NoopReset -> MaxAndSkip -> EpisodicLife -> FireReset
            -> ClipReward -> ResizeAndGrayscale -> FrameStack
"""

from __future__ import annotations

import collections
from typing import Any, SupportsFloat

import ale_py  # noqa: F401  — registers ALE environments
import cv2
import gymnasium
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# 1. NoopResetEnv
# ---------------------------------------------------------------------------

class NoopResetEnv(gymnasium.Wrapper):
    """Execute a random number of no-ops on reset to randomise the start state.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to wrap.
    noop_max : int
        Upper bound (inclusive) on the number of no-op actions executed after
        a reset.  The lower bound is 1, so at least one no-op is always sent.
    """

    def __init__(self, env: gymnasium.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP is action 0 in all standard Atari envs

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        noop_count = self.np_random.integers(1, self.noop_max + 1)
        for _ in range(noop_count):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


# ---------------------------------------------------------------------------
# 2. MaxAndSkipEnv
# ---------------------------------------------------------------------------

class MaxAndSkipEnv(gymnasium.Wrapper):
    """Repeat the chosen action for *skip* frames and return the pixel-wise
    maximum of the last two frames (removes Atari sprite flicker).

    Rewards are accumulated over the skipped frames.
    """

    def __init__(self, env: gymnasium.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        assert env.observation_space.shape is not None
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=np.uint8
        )

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Store the last two observations for max pooling.
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# 3. EpisodicLifeEnv
# ---------------------------------------------------------------------------

class EpisodicLifeEnv(gymnasium.Wrapper):
    """Signal episode end (``terminated=True``) whenever a life is lost.

    This provides a denser training signal without actually resetting the
    underlying environment.  The *real* episode termination is forwarded on the
    first step of the next "pseudo-episode", triggering the proper reset.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated

        lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        if 0 < lives < self.lives:
            # Life was lost -- mark as terminated for training purposes.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.was_real_done:
            obs, info = self.env.reset(seed=seed, options=options)
        else:
            # The underlying env is still running; take a no-op to advance
            # past the life-loss frame.
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()  # type: ignore[attr-defined]
        return obs, info


# ---------------------------------------------------------------------------
# 4. FireResetEnv
# ---------------------------------------------------------------------------

class FireResetEnv(gymnasium.Wrapper):
    """Automatically press FIRE after reset for environments that require it
    (e.g. Breakout).
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.env.reset(seed=seed, options=options)
        # FIRE (action 1) to start the game.
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset()
        # Some envs also need a second step after FIRE.
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, info


# ---------------------------------------------------------------------------
# 5. ClipRewardEnv
# ---------------------------------------------------------------------------

class ClipRewardEnv(gymnasium.Wrapper):
    """Clip rewards to {-1, 0, +1} using ``np.sign``."""

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(np.sign(reward)), terminated, truncated, info


# ---------------------------------------------------------------------------
# 6. ResizeAndGrayscale
# ---------------------------------------------------------------------------

class ResizeAndGrayscale(gymnasium.ObservationWrapper):
    """Resize observations to 84x84 and convert to grayscale.

    Output observation space: ``Box(0, 255, (84, 84, 1), uint8)``.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Convert RGB -> grayscale.
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84.
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized[:, :, np.newaxis]  # (84, 84, 1)


# ---------------------------------------------------------------------------
# 7. FrameStack
# ---------------------------------------------------------------------------

class FrameStack(gymnasium.Wrapper):
    """Stack the last *k* frames along the last axis using a
    ``collections.deque``.

    Output observation space: ``Box(0, 255, (84, 84, k), uint8)``.
    """

    def __init__(self, env: gymnasium.Env, k: int = 4) -> None:
        super().__init__(env)
        self.k = k
        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=k)
        low = np.repeat(env.observation_space.low, k, axis=-1)
        high = np.repeat(env.observation_space.high, k, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.uint8
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        assert len(self._frames) == self.k
        return np.concatenate(list(self._frames), axis=-1)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_atari_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    training: bool = True,
) -> gymnasium.Env:
    """Create an Atari environment with the full standard preprocessing stack.

    Suitable for use as a factory callable with
    ``gymnasium.vector.SyncVectorEnv``::

        vec_env = gymnasium.vector.SyncVectorEnv([
            lambda: make_atari_env("ALE/Breakout-v5", seed=i)
            for i in range(8)
        ])

    Parameters
    ----------
    env_id : str
        Gymnasium environment identifier.
    seed : int | None
        Optional seed for the environment's RNG.
    render_mode : str | None
        Gymnasium render mode (``"human"``, ``"rgb_array"``, etc.).
    training : bool
        If True, include ``EpisodicLifeEnv`` (end episode on life loss)
        and ``ClipRewardEnv`` (clip rewards to {-1, 0, +1}).  Set to
        False for evaluation so that full-game episodes and raw rewards
        are used.

    Returns
    -------
    gymnasium.Env
        The fully wrapped Atari environment.
    """
    env = gymnasium.make(env_id, render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if training:
        env = EpisodicLifeEnv(env)

    # Only apply FireReset when the env actually has a FIRE action.
    if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
        env = FireResetEnv(env)

    if training:
        env = ClipRewardEnv(env)

    env = ResizeAndGrayscale(env)
    env = FrameStack(env, k=4)

    if seed is not None:
        env.reset(seed=seed)

    return env
