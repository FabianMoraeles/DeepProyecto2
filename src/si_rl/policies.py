"""Politicas de inferencia y carga de checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from si_rl.envs import EnvConfig
from si_rl.networks import build_network


class GreedyPolicy:
    """argmax_a Q(s, a), con epsilon opcional para evaluacion estocastica."""

    def __init__(self, net: torch.nn.Module, device: torch.device,
                 epsilon: float = 0.0, seed: int = 0):
        self.net = net
        self.device = device
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.net.eval()

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> int:
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.net.n_actions))
        t = torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0)
        return int(self.net(t).argmax(dim=1).item())

    @torch.no_grad()
    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """Version vectorizada: obs (N, C, H, W) -> acciones (N,)."""
        t = torch.as_tensor(np.asarray(obs), device=self.device)
        acciones = self.net(t).argmax(dim=1).cpu().numpy()
        if self.epsilon > 0.0:
            aleatorias = self.rng.random(acciones.shape[0]) < self.epsilon
            if aleatorias.any():
                acciones[aleatorias] = self.rng.integers(
                    self.net.n_actions, size=int(aleatorias.sum())
                )
        return acciones


def save_checkpoint(path: str | Path, *, net: torch.nn.Module, env_config: EnvConfig,
                    train_config: dict, step: int, extra: dict | None = None,
                    arch: str = "dqn", net_kwargs: dict | None = None,
                    optimizer: torch.optim.Optimizer | None = None) -> None:
    """Guarda pesos, arquitectura, configuracion del entorno y metadatos juntos.

    El EnvConfig y los argumentos de la red viajan con los pesos para que sea
    imposible evaluar un checkpoint con un preprocesamiento o una arquitectura
    distintos a los que se entrenaron.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "env_config": env_config.to_dict(),
            "arch": arch,
            "net_kwargs": net_kwargs or {
                "in_channels": env_config.obs_shape[0],
                "n_actions": net.n_actions,
                "dueling": getattr(net, "dueling", False),
            },
            "train_config": train_config,
            "step": step,
            "n_actions": net.n_actions,
            # Permite reanudar el entrenamiento sin reiniciar el estado de Adam.
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "extra": extra or {},
        },
        path,
    )


def load_agent(path: str | Path, device: str | torch.device = "cuda",
               epsilon: float = 0.0, seed: int = 0) -> tuple[GreedyPolicy, EnvConfig, dict]:
    """Carga un checkpoint y devuelve (politica, EnvConfig usado, metadatos)."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    env_config = EnvConfig(**ckpt["env_config"])
    net_kwargs = ckpt.get("net_kwargs") or {
        "in_channels": env_config.obs_shape[0],
        "n_actions": ckpt["n_actions"],
        "dueling": False,
    }
    net = build_network(ckpt.get("arch", "dqn"), **net_kwargs).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    meta = {
        "step": ckpt["step"],
        "train_config": ckpt["train_config"],
        "extra": ckpt["extra"],
    }
    return GreedyPolicy(net, device, epsilon=epsilon, seed=seed), env_config, meta
