"""Offline, dataset-driven environment for training the DDPG detector.

Each step presents one labelled KPM sample as the observation. The agent
outputs a continuous action in ``[-1, 1]``; its first component is mapped
to a jamming score ``p = (a + 1) / 2``. The reward is

    r = w_c * (1 - 2 * |p - target|)

where ``target`` is 1 for jamming and 0 for normal traffic and ``w_c`` is an
inverse-frequency class weight, so that the rare class is not ignored. The
reward is in ``[-w_c, w_c]``: fully confident correct answers earn ``+w_c``,
fully confident wrong answers ``-w_c``.

Samples are independent, so the problem is a contextual bandit; the agent
is trained with ``gamma = 0`` by default.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from jamming_detection.data.loader import NORMAL_LABEL


class JammingDatasetEnv(gym.Env):
    """Gymnasium environment that replays a labelled feature matrix."""

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        episode_length: Optional[int] = None,
        action_dim: int = 1,
        normal_label: str = NORMAL_LABEL,
        class_weighted: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.X = np.asarray(X, dtype=np.float32)
        self.targets = (np.asarray(y) != normal_label).astype(np.float32)
        self.episode_length = episode_length or len(self.X)

        pos = float(self.targets.mean())
        if class_weighted and 0.0 < pos < 1.0:
            self.class_weights = {1.0: 0.5 / pos, 0.0: 0.5 / (1.0 - pos)}
        else:
            self.class_weights = {1.0: 1.0, 0.0: 1.0}

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(self.X))
        self._ptr = 0
        self._step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(len(self.X))
        self._ptr = 0
        self._step = 0
        return self.X[self._order[0]], {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        idx = self._order[self._ptr]
        target = float(self.targets[idx])
        score = float(np.clip((np.asarray(action).ravel()[0] + 1.0) / 2.0, 0.0, 1.0))
        reward = self.class_weights[target] * (1.0 - 2.0 * abs(score - target))

        self._ptr += 1
        self._step += 1
        terminated = self._ptr >= len(self._order)
        truncated = self._step >= self.episode_length
        obs = self.X[self._order[self._ptr]] if not terminated else np.zeros_like(self.X[0])
        info = {"target": target, "score": score, "correct": (score > 0.5) == bool(target)}
        return obs, float(reward), terminated, truncated, info
