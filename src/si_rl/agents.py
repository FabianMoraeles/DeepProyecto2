"""Agentes: encapsulan seleccion de accion y calculo de perdida.

El bucle de entrenamiento (entorno, replay, logging, checkpoints, evaluacion) es
identico para DQN e IQN, asi que vive una sola vez en train_dqn.py y lo unico
que cambia por algoritmo esta aqui. Eso ademas garantiza que las ablaciones se
comparen bajo exactamente el mismo bucle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from si_rl.networks import build_network
from si_rl.replay import Batch


H_EPS = 1e-2


def h(x: torch.Tensor, eps: float = H_EPS) -> torch.Tensor:
    """Reescalado invertible de valores (Pohlen et al., 2018).

        h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x

    Alternativa al reward clipping. Clipar a signo hace que un alien de 5 puntos,
    uno de 30 y la nave nodriza de 200 valgan exactamente lo mismo, asi que el
    agente no tiene forma de saber que la nodriza vale 40 veces mas. El
    reescalado conserva el orden de magnitud de las recompensas y a la vez
    comprime el rango de Q, que es lo que hacia inestable entrenar sin clipar.
    """
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def h_inv(x: torch.Tensor, eps: float = H_EPS) -> torch.Tensor:
    """Inversa de `h`."""
    interior = 1.0 + 4.0 * eps * (torch.abs(x) + 1.0 + eps)
    return torch.sign(x) * (((torch.sqrt(interior) - 1.0) / (2.0 * eps)) ** 2 - 1.0)


@dataclass
class ResultadoUpdate:
    perdida: float
    q_medio: float
    prioridades: np.ndarray    # senal para PER (|error TD| o su equivalente)


class BaseAgent:
    """Interfaz comun. `net` es siempre utilizable como Q(s,.) -> (B, A)."""

    net: torch.nn.Module
    target: torch.nn.Module
    opt: torch.optim.Optimizer

    def __init__(self, cfg, env_cfg, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = env_cfg.n_actions
        self.net_kwargs = self._construir_kwargs(env_cfg)
        self.net = build_network(self.arch, **self.net_kwargs).to(device)
        self.target = build_network(self.arch, **self.net_kwargs).to(device)
        self.target.load_state_dict(self.net.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    def _construir_kwargs(self, env_cfg) -> dict:
        return dict(in_channels=env_cfg.obs_shape[0], n_actions=env_cfg.n_actions,
                    dueling=self.cfg.dueling)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        q = self.net(torch.as_tensor(obs, device=self.device))
        acciones = q.float().argmax(dim=1).cpu().numpy()
        explorar = rng.random(acciones.shape[0]) < epsilon
        if explorar.any():
            acciones[explorar] = rng.integers(0, self.n_actions, size=int(explorar.sum()))
        return acciones

    def _tensores(self, lote: Batch):
        d = self.device
        return (
            torch.as_tensor(lote.obs, device=d),
            torch.as_tensor(lote.next_obs, device=d),
            torch.as_tensor(lote.actions, device=d),
            torch.as_tensor(lote.returns, device=d),
            torch.as_tensor(lote.done, device=d),
            torch.as_tensor(lote.n_steps, device=d).float(),
        )

    def _paso_optimizador(self, perdida: torch.Tensor) -> None:
        self.opt.zero_grad(set_to_none=True)
        perdida.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

    def _pesos(self, lote: Batch) -> torch.Tensor | None:
        if not self.cfg.per or lote.weights is None:
            return None
        return torch.as_tensor(lote.weights, device=self.device)


class DQNAgent(BaseAgent):
    arch = "dqn"

    def update(self, lote: Batch) -> ResultadoUpdate:
        obs_b, next_b, acc_b, ret_b, done_b, n_b = self._tensores(lote)
        gamma = self.cfg.gamma

        with torch.no_grad():
            if self.cfg.double_dqn:
                # La red en linea elige la accion y la objetivo la valua: separar
                # ambas cosas quita el sesgo optimista de tomar el max sobre
                # valores ruidosos.
                mejor = self.net(next_b).argmax(dim=1, keepdim=True)
                q_next = self.target(next_b).gather(1, mejor).squeeze(1)
            else:
                q_next = self.target(next_b).max(dim=1).values

            if self.cfg.reward_transform == "h":
                # La red opera en espacio h; hay que volver al espacio real para
                # sumar la recompensa y reescalar el resultado.
                objetivo = h(ret_b + (gamma ** n_b) * h_inv(q_next) * (~done_b))
            else:
                objetivo = ret_b + (gamma ** n_b) * q_next * (~done_b)

        q_pred = self.net(obs_b).gather(1, acc_b.unsqueeze(1)).squeeze(1)
        td = q_pred - objetivo
        por_muestra = F.smooth_l1_loss(q_pred, objetivo, reduction="none")

        pesos = self._pesos(lote)
        perdida = (por_muestra * pesos).mean() if pesos is not None else por_muestra.mean()
        self._paso_optimizador(perdida)

        return ResultadoUpdate(
            perdida=float(perdida.detach()),
            q_medio=float(q_pred.detach().mean()),
            prioridades=td.detach().abs().cpu().numpy(),
        )


class IQNAgent(BaseAgent):
    arch = "iqn"

    def _construir_kwargs(self, env_cfg) -> dict:
        return dict(in_channels=env_cfg.obs_shape[0], n_actions=env_cfg.n_actions,
                    dueling=self.cfg.dueling, n_cos=self.cfg.iqn_n_cos,
                    k_eval=self.cfg.iqn_k)

    def update(self, lote: Batch) -> ResultadoUpdate:
        obs_b, next_b, acc_b, ret_b, done_b, n_b = self._tensores(lote)
        cfg = self.cfg
        b = obs_b.shape[0]
        n, n_p, kappa = cfg.iqn_n, cfg.iqn_n_prime, cfg.iqn_kappa

        with torch.no_grad():
            elector = self.net if cfg.double_dqn else self.target
            mejor = elector(next_b, k=cfg.iqn_k).argmax(dim=1)          # (B,)
            taus_p = torch.rand(b, n_p, device=self.device)
            z_next = self.target.quantiles(next_b, taus_p)              # (B, N', A)
            z_next = z_next.gather(2, mejor.view(b, 1, 1).expand(b, n_p, 1)).squeeze(2)
            gamma_n = (cfg.gamma ** n_b).unsqueeze(1)
            vivo = (~done_b).float().unsqueeze(1)
            if cfg.reward_transform == "h":
                objetivo = h(ret_b.unsqueeze(1) + gamma_n * h_inv(z_next) * vivo)
            else:
                objetivo = ret_b.unsqueeze(1) + gamma_n * z_next * vivo   # (B, N')

        taus = torch.rand(b, n, device=self.device)
        z = self.net.quantiles(obs_b, taus)                             # (B, N, A)
        z = z.gather(2, acc_b.view(b, 1, 1).expand(b, n, 1)).squeeze(2)  # (B, N)

        # Perdida de Huber cuantilica: cada cuantil predicho se compara contra
        # todas las muestras del objetivo, y el peso |tau - 1(td<0)| es lo que
        # hace que el cuantil tau converja al cuantil tau de la distribucion.
        td = objetivo.unsqueeze(1) - z.unsqueeze(2)                     # (B, N, N')
        abs_td = td.abs()
        huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2),
                            kappa * (abs_td - 0.5 * kappa))
        rho = (taus.unsqueeze(2) - (td.detach() < 0).float()).abs() * huber / kappa
        por_muestra = rho.sum(dim=1).mean(dim=1)                        # (B,)

        pesos = self._pesos(lote)
        perdida = (por_muestra * pesos).mean() if pesos is not None else por_muestra.mean()
        self._paso_optimizador(perdida)

        return ResultadoUpdate(
            perdida=float(perdida.detach()),
            q_medio=float(z.detach().mean()),
            prioridades=por_muestra.detach().cpu().numpy(),
        )


def make_agent(cfg, env_cfg, device: torch.device) -> BaseAgent:
    if cfg.algo == "dqn":
        return DQNAgent(cfg, env_cfg, device)
    if cfg.algo == "iqn":
        return IQNAgent(cfg, env_cfg, device)
    raise ValueError(f"algoritmo desconocido: {cfg.algo}")
