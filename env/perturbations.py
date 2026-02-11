"""Visual perturbation wrappers for evaluating agent robustness.

These wrappers inject controlled visual noise into Atari observations so that
a trained agent can be stress-tested at evaluation time.  They operate on
**raw RGB** frames and must therefore be inserted *before* the
``ResizeAndGrayscale`` wrapper in the preprocessing pipeline.

Perturbation levels (used by ``make_perturbed_atari_env``):

- ``"clean"`` -- no perturbation at all
- ``"mild"``  -- subtle colour jitter + light Gaussian noise
- ``"hard"``  -- aggressive colour jitter + heavy Gaussian noise
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import cv2
import gymnasium
import numpy as np

from env.wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    FrameStack,
    MaxAndSkipEnv,
    NoopResetEnv,
    ResizeAndGrayscale,
)


# ---------------------------------------------------------------------------
# 1. ColorJitterWrapper
# ---------------------------------------------------------------------------

class ColorJitterWrapper(gymnasium.ObservationWrapper):
    """Apply random but episode-consistent colour jitter to RGB frames.

    On each ``reset`` new jitter parameters are sampled; they are then applied
    identically to every frame within the episode.

    Parameters
    ----------
    env : gymnasium.Env
        Wrapped environment (must emit RGB uint8 observations).
    hue_range : float
        Maximum absolute hue shift (fraction of the [0, 180) hue circle used
        by OpenCV).  The shift is sampled uniformly from
        ``[-hue_range * 180, hue_range * 180]``.
    sat_range : float
        Saturation is scaled by a factor sampled uniformly from
        ``[1 - sat_range, 1 + sat_range]``.
    bright_range : float
        Brightness offset sampled uniformly from
        ``[-bright_range * 255, bright_range * 255]``.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        hue_range: float = 0.2,
        sat_range: float = 0.3,
        bright_range: float = 0.3,
    ) -> None:
        super().__init__(env)
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.bright_range = bright_range

        # Per-episode jitter parameters (set on reset).
        self._hue_shift: float = 0.0
        self._sat_scale: float = 1.0
        self._bright_offset: float = 0.0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        # Sample new episode-level jitter parameters.
        rng = self.np_random
        self._hue_shift = float(
            rng.uniform(-self.hue_range * 180, self.hue_range * 180)
        )
        self._sat_scale = float(
            rng.uniform(1.0 - self.sat_range, 1.0 + self.sat_range)
        )
        self._bright_offset = float(
            rng.uniform(-self.bright_range * 255, self.bright_range * 255)
        )

        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Convert RGB -> HSV (OpenCV uses H in [0,180), S/V in [0,255]).
        hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue: shift and wrap around [0, 180).
        hsv[:, :, 0] = (hsv[:, :, 0] + self._hue_shift) % 180.0

        # Saturation: scale.
        hsv[:, :, 1] = hsv[:, :, 1] * self._sat_scale

        # Value (brightness): offset.
        hsv[:, :, 2] = hsv[:, :, 2] + self._bright_offset

        # Clip to valid ranges and convert back.
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)

        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb


# ---------------------------------------------------------------------------
# 2. GaussianNoiseWrapper
# ---------------------------------------------------------------------------

class GaussianNoiseWrapper(gymnasium.ObservationWrapper):
    """Add independent Gaussian noise to every pixel on every step.

    Parameters
    ----------
    env : gymnasium.Env
        Wrapped environment (must emit uint8 observations).
    std : float
        Standard deviation of the Gaussian noise (in pixel intensity units,
        i.e. on the [0, 255] scale).
    """

    def __init__(self, env: gymnasium.Env, std: float = 15.0) -> None:
        super().__init__(env)
        self.std = std

    def observation(self, obs: np.ndarray) -> np.ndarray:
        noise = self.np_random.normal(0.0, self.std, size=obs.shape)
        noisy = np.clip(obs.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)


# ---------------------------------------------------------------------------
# Perturbation level presets
# ---------------------------------------------------------------------------

_PERTURBATION_PRESETS: dict[str, dict[str, Any]] = {
    "clean": {},
    "color_jitter": {
        "color_jitter": {"hue_range": 0.2, "sat_range": 0.3, "bright_range": 0.2},
    },
    "noise": {
        "gaussian_noise": {"std": 10.0},
    },
    "mild": {
        "color_jitter": {"hue_range": 0.2, "sat_range": 0.3, "bright_range": 0.2},
        "gaussian_noise": {"std": 10.0},
    },
    "hard": {
        "color_jitter": {"hue_range": 0.4, "sat_range": 0.5, "bright_range": 0.4},
        "gaussian_noise": {"std": 25.0},
    },
}


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_perturbed_atari_env(
    env_id: str,
    perturbation_level: str = "clean",
    seed: int | None = None,
    render_mode: str | None = None,
    training: bool = False,
) -> gymnasium.Env:
    """Create an Atari environment with optional visual perturbations.

    The wrapper stack is applied in the following order:

    1. Raw Atari env (``gymnasium.make``)
    2. ``NoopResetEnv``
    3. ``MaxAndSkipEnv``
    4. ``EpisodicLifeEnv`` (training only)
    5. ``FireResetEnv`` (if the env requires FIRE)
    6. ``ClipRewardEnv`` (training only)
    7. **Perturbation wrappers** (colour jitter, then Gaussian noise)
    8. ``ResizeAndGrayscale``
    9. ``FrameStack``

    Perturbation wrappers are placed *after* reward-related wrappers but
    *before* the grayscale conversion so that colour jitter has the intended
    effect.

    Parameters
    ----------
    env_id : str
        Gymnasium environment identifier.
    perturbation_level : str
        One of ``"clean"``, ``"color_jitter"``, ``"noise"``,
        ``"mild"``, or ``"hard"``.
    seed : int | None
        Optional seed for the environment's RNG.
    render_mode : str | None
        Gymnasium render mode.
    training : bool
        If True, include ``EpisodicLifeEnv`` and ``ClipRewardEnv``.
        Default False (evaluation mode).

    Returns
    -------
    gymnasium.Env
        The fully wrapped (and optionally perturbed) Atari environment.
    """
    if perturbation_level not in _PERTURBATION_PRESETS:
        raise ValueError(
            f"Unknown perturbation_level {perturbation_level!r}. "
            f"Choose from {list(_PERTURBATION_PRESETS)}."
        )

    preset = _PERTURBATION_PRESETS[perturbation_level]

    # -- 1. Raw env --------------------------------------------------------
    env = gymnasium.make(env_id, render_mode=render_mode)

    # -- 2-6. Standard Atari wrappers (reward / life / skip) ---------------
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    if training:
        env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():  # type: ignore[attr-defined]
        env = FireResetEnv(env)

    if training:
        env = ClipRewardEnv(env)

    # -- 7. Visual perturbation wrappers (before grayscale) ----------------
    if "color_jitter" in preset:
        env = ColorJitterWrapper(env, **preset["color_jitter"])
    if "gaussian_noise" in preset:
        env = GaussianNoiseWrapper(env, **preset["gaussian_noise"])

    # -- 8-9. Observation processing ---------------------------------------
    env = ResizeAndGrayscale(env)
    env = FrameStack(env, k=4)

    if seed is not None:
        env.reset(seed=seed)

    return env
