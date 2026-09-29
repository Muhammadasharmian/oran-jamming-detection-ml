"""Deep Deterministic Policy Gradient (DDPG) jamming detector.

Three actor variants are provided:

``mlp``
    Fully connected policy on the numeric KPM state.
``llm``
    Frozen DistilBERT encoder over a textual description of the state,
    followed by a trainable policy head.
``hybrid``
    Concatenation of an MLP branch on the numeric state and a DistilBERT
    branch on the textual description.

The ``llm`` and ``hybrid`` actors require the optional ``transformers``
package and download ``distilbert-base-uncased`` on first use.

:class:`DDPGDetector` wraps the agent and :class:`JammingDatasetEnv` in a
scikit-learn style ``fit`` / ``predict`` interface so it can be trained and
evaluated alongside the other models.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin

from jamming_detection import config
from jamming_detection.data.loader import JAMMING_LABEL, NORMAL_LABEL
from jamming_detection.envs.dataset_env import JammingDatasetEnv

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return map(np.stack, zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """Temporally correlated exploration noise."""

    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2) -> None:
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.normal(size=self.state.shape)
        self.state = self.state + dx
        return self.state


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.state_fc = nn.Linear(state_dim, hidden_dims[0])
        self.action_fc = nn.Linear(action_dim, hidden_dims[0])
        self.hidden = nn.ModuleList(
            nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)
        )
        self.out = nn.Linear(hidden_dims[-1], 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.state_fc(state)) + F.relu(self.action_fc(action))
        for layer in self.hidden:
            x = self.dropout(F.relu(layer(x)))
        return self.out(x)


def _mlp(in_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1) -> nn.Sequential:
    layers: List[nn.Module] = []
    for width in hidden_dims:
        layers += [nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = width
    return nn.Sequential(*layers)


class MLPActor(nn.Module):
    uses_text = False

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.body = _mlp(state_dim, hidden_dims)
        self.head = nn.Sequential(nn.Linear(hidden_dims[-1], action_dim), nn.Tanh())

    def forward(self, state: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        return self.head(self.body(state))


class _TextEncoder(nn.Module):
    """Frozen DistilBERT sentence encoder (mean pooled)."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        try:
            from transformers import DistilBertModel, DistilBertTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("The 'llm' and 'hybrid' actors require `pip install transformers`.") from exc
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.encoder = DistilBertModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.output_dim = self.encoder.config.hidden_size

    def forward(self, prompts: List[str]) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            hidden = self.encoder(**tokens).last_hidden_state
        return hidden.mean(dim=1)


class LLMActor(nn.Module):
    uses_text = True

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.text = _TextEncoder()
        self.body = _mlp(self.text.output_dim, hidden_dims)
        self.head = nn.Sequential(nn.Linear(hidden_dims[-1], action_dim), nn.Tanh())

    def forward(self, state: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        return self.head(self.body(self.text(prompts)))


class HybridActor(nn.Module):
    uses_text = True

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        self.text = _TextEncoder()
        self.state_branch = _mlp(state_dim, hidden_dims)
        self.text_branch = _mlp(self.text.output_dim, hidden_dims)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, prompts: Optional[List[str]] = None) -> torch.Tensor:
        fused = torch.cat([self.state_branch(state), self.text_branch(self.text(prompts))], dim=-1)
        return self.fusion(fused)


_ACTORS = {"mlp": MLPActor, "llm": LLMActor, "hybrid": HybridActor}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DDPGAgent:
    """DDPG agent with target networks, replay buffer and OU exploration."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        actor_type: str = "mlp",
        params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
    ) -> None:
        if actor_type not in _ACTORS:
            raise ValueError(f"Unknown actor type '{actor_type}'. Choose from {list(_ACTORS)}")
        self.cfg = {**config.DDPG, **(params or {})}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_type = actor_type
        self.feature_names = list(feature_names) if feature_names else [f"f{i}" for i in range(state_dim)]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        actor_cls = _ACTORS[actor_type]
        hidden = tuple(self.cfg["hidden_dims"])
        actor_hidden = hidden if actor_type == "mlp" else None
        kwargs = {"hidden_dims": actor_hidden} if actor_hidden else {}
        self.actor = actor_cls(state_dim, action_dim, **kwargs).to(self.device)
        self.actor_target = actor_cls(state_dim, action_dim, **kwargs).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_opt = torch.optim.Adam(trainable, lr=self.cfg["actor_lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg["critic_lr"])

        self.buffer = ReplayBuffer(self.cfg["replay_buffer_size"])
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim, theta=self.cfg["noise_theta"], sigma=self.cfg["noise_sigma"]
        )
        self.history: Dict[str, List[float]] = {"episode_reward": [], "critic_loss": [], "actor_loss": []}

    # -- helpers ------------------------------------------------------------

    def _prompts(self, states: np.ndarray) -> Optional[List[str]]:
        """Textual state description for the language-model actors."""
        if not getattr(self.actor, "uses_text", False):
            return None
        return [
            "Radio link report: "
            + ", ".join(f"{n} is {v:.3f}" for n, v in zip(self.feature_names, row))
            + ". Decide whether the link is being jammed."
            for row in np.atleast_2d(states)
        ]

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for t, s in zip(target.parameters(), source.parameters()):
                t.mul_(1.0 - tau).add_(tau * s)

    # -- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        self.actor.eval()
        s = torch.as_tensor(np.atleast_2d(state), dtype=torch.float32, device=self.device)
        action = self.actor(s, self._prompts(state)).cpu().numpy()
        self.actor.train()
        if explore:
            action = np.clip(action + self.noise.sample(), -1.0, 1.0)
        return action[0] if np.ndim(state) == 1 else action

    # -- learning -----------------------------------------------------------

    def update(self) -> None:
        batch_size = self.cfg["batch_size"]
        if len(self.buffer) < batch_size:
            return
        s, a, r, s2, d = self.buffer.sample(batch_size)
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)  # noqa: E731
        s_t, a_t, s2_t = to_t(s), to_t(a), to_t(s2)
        r_t, d_t = to_t(r).unsqueeze(1), to_t(d).unsqueeze(1)

        with torch.no_grad():
            next_a = self.actor_target(s2_t, self._prompts(s2))
            target_q = r_t + (1.0 - d_t) * self.cfg["gamma"] * self.critic_target(s2_t, next_a)
        critic_loss = F.mse_loss(self.critic(s_t, a_t), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        actor_loss = -self.critic(s_t, self.actor(s_t, self._prompts(s))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor, self.cfg["tau"])
        self._soft_update(self.critic_target, self.critic, self.cfg["tau"])
        self.history["critic_loss"].append(critic_loss.item())
        self.history["actor_loss"].append(actor_loss.item())

    def train(
        self,
        env: JammingDatasetEnv,
        total_steps: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        total_steps = total_steps or self.cfg["total_steps"]
        clip = self.cfg.get("reward_clip")
        warmup = self.cfg["warmup_steps"]
        state, _ = env.reset(seed=seed)
        self.noise.reset()
        ep_reward, step = 0.0, 0
        while step < total_steps:
            if step < warmup:
                action = env.action_space.sample()
            else:
                action = self.act(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if clip:
                reward = float(np.clip(reward, -clip, clip))
            self.buffer.push(state, action, reward, next_state, float(terminated))
            if step >= warmup:
                self.update()
            ep_reward += reward
            state = next_state
            step += 1
            if terminated or truncated:
                self.history["episode_reward"].append(ep_reward)
                if verbose:
                    n = len(self.history["episode_reward"])
                    print(f"  episode {n:4d}  step {step:7d}  reward {ep_reward:9.2f}")
                state, _ = env.reset()
                self.noise.reset()
                ep_reward = 0.0
        return self.history

    # -- persistence --------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "cfg": self.cfg,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "actor_type": self.actor_type,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], device: Optional[str] = None) -> DDPGAgent:
        agent = cls(
            state["state_dim"],
            state["action_dim"],
            state["actor_type"],
            state["cfg"],
            state["feature_names"],
            device,
        )
        agent.actor.load_state_dict(state["actor"])
        agent.critic.load_state_dict(state["critic"])
        agent.actor_target.load_state_dict(state["actor"])
        agent.critic_target.load_state_dict(state["critic"])
        return agent


# ---------------------------------------------------------------------------
# Scikit-learn style wrapper
# ---------------------------------------------------------------------------


class DDPGDetector(BaseEstimator, ClassifierMixin):
    """Binary jamming detector trained with DDPG on :class:`JammingDatasetEnv`."""

    def __init__(
        self,
        actor_type: str = "mlp",
        params: Optional[Dict[str, Any]] = None,
        total_steps: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        normal_label: str = NORMAL_LABEL,
        random_state: int = config.RANDOM_STATE,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.actor_type = actor_type
        self.params = params
        self.total_steps = total_steps
        self.feature_names = feature_names
        self.normal_label = normal_label
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> DDPGDetector:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        cfg = {**config.DDPG, **(self.params or {})}
        env = JammingDatasetEnv(
            X, y, episode_length=cfg["episode_length"], normal_label=self.normal_label, seed=self.random_state
        )
        self.agent_ = DDPGAgent(X.shape[1], 1, self.actor_type, cfg, self.feature_names, self.device)
        self.agent_.train(
            env, self.total_steps or cfg["total_steps"], seed=self.random_state, verbose=self.verbose
        )
        self.classes_ = np.array([JAMMING_LABEL, self.normal_label])
        return self

    def jamming_probability(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = [self.agent_.act(X[i : i + 1024])[:, 0] for i in range(0, len(X), 1024)]
        return (np.concatenate(out) + 1.0) / 2.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.jamming_probability(X)
        return np.column_stack([p, 1.0 - p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.jamming_probability(X) > 0.5, JAMMING_LABEL, self.normal_label)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if "agent_" in state:
            state["agent_"] = state["agent_"].state_dict()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        agent = state.get("agent_")
        if isinstance(agent, dict):
            state["agent_"] = DDPGAgent.from_state_dict(agent, device="cpu")
        self.__dict__.update(state)
