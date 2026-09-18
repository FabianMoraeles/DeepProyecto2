"""Arquitecturas de red.

La normalizacion uint8 -> [0,1] vive dentro del modulo a proposito: asi es
imposible olvidarla en evaluacion y que el agente vea una escala distinta a la
de entrenamiento.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def nature_encoder(in_channels: int) -> nn.Sequential:
    """CNN de Mnih et al. (2015). Con entrada (C, 84, 84) produce 3136 features."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(inplace=True),
        nn.Flatten(),
    )


FEATURES = 3136


class QNetwork(nn.Module):
    """Q(s, .) sobre la CNN de Nature, con cabeza lineal o dueling.

    En la variante dueling la red estima por separado el valor del estado y la
    ventaja de cada accion:

        Q(s, a) = V(s) + A(s, a) - media_a A(s, a)

    Restar la media fija la indeterminacion entre V y A (sumar una constante a V
    y restarla a A daria el mismo Q), que si no haria el entrenamiento inestable.
    Ayuda en Space Invaders porque en la mayoria de estados da casi igual que
    accion tomar, y separar V de A permite aprender el valor del estado sin
    tener que estimar bien las seis acciones.
    """

    def __init__(self, in_channels: int = 4, n_actions: int = 6,
                 hidden: int = 512, dueling: bool = False):
        super().__init__()
        self.n_actions = n_actions
        self.dueling = dueling
        self.encoder = nature_encoder(in_channels)

        if dueling:
            self.valor = nn.Sequential(
                nn.Linear(FEATURES, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
            )
            self.ventaja = nn.Sequential(
                nn.Linear(FEATURES, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n_actions),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(FEATURES, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n_actions),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W) uint8 o float. Devuelve Q(s, .) de shape (B, n_actions)."""
        x = obs.float().div_(255.0) if obs.dtype == torch.uint8 else obs
        z = self.encoder(x)
        if not self.dueling:
            return self.head(z)
        v = self.valor(z)
        a = self.ventaja(z)
        return v + a - a.mean(dim=1, keepdim=True)


class IQNNetwork(nn.Module):
    """Implicit Quantile Network (Dabney et al., 2018).

    En vez de estimar el valor esperado Q(s,a), estima la funcion cuantil
    Z_tau(s,a) de la distribucion de retornos. Aprender la distribucion completa
    da una senal de aprendizaje mucho mas rica que su media, y en Space Invaders
    es donde mas se nota: los numeros publicados a 200M frames son ~5,699 para
    Rainbow contra ~28,888 para IQN.

    El cuantil tau entra por un embedding de cosenos que se multiplica elemento a
    elemento con las features de la CNN, de modo que una sola red representa
    todos los cuantiles.

    `forward` devuelve Q(s,.) promediando K cuantiles, para que la red se pueda
    usar igual que una QNetwork al actuar y al evaluar. `quantiles` expone los
    cuantiles crudos, que es lo que necesita la perdida.
    """

    def __init__(self, in_channels: int = 4, n_actions: int = 6, hidden: int = 512,
                 n_cos: int = 64, k_eval: int = 32, dueling: bool = False):
        super().__init__()
        self.n_actions = n_actions
        self.n_cos = n_cos
        self.k_eval = k_eval
        self.dueling = dueling
        # Distorsion de riesgo, solo para evaluacion. Se deja neutra al entrenar.
        self.distortion: str = "neutral"
        self.distortion_eta: float = 0.71
        self.encoder = nature_encoder(in_channels)
        self.cos_embed = nn.Linear(n_cos, FEATURES)

        if dueling:
            self.valor = nn.Sequential(
                nn.Linear(FEATURES, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1)
            )
            self.ventaja = nn.Sequential(
                nn.Linear(FEATURES, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n_actions),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(FEATURES, hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, n_actions),
            )
        self.register_buffer(
            "_rango", torch.arange(1, n_cos + 1, dtype=torch.float32).view(1, 1, n_cos)
        )

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().div_(255.0) if obs.dtype == torch.uint8 else obs
        return self.encoder(x)

    def quantiles(self, obs: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """obs: (B, C, H, W); taus: (B, N). Devuelve Z_tau(s, .) de shape (B, N, A)."""
        z = self._features(obs)                                  # (B, F)
        cos = torch.cos(taus.unsqueeze(-1) * self._rango * torch.pi)   # (B, N, n_cos)
        phi = torch.relu(self.cos_embed(cos))                    # (B, N, F)
        h = z.unsqueeze(1) * phi                                 # (B, N, F)
        if not self.dueling:
            return self.head(h)
        v = self.valor(h)
        a = self.ventaja(h)
        return v + a - a.mean(dim=2, keepdim=True)

    def distorsionar(self, taus: torch.Tensor) -> torch.Tensor:
        """Aplica beta(tau) para obtener una valuacion sensible al riesgo.

        Q_beta(s,a) = E_{tau ~ U(0,1)}[ Z_{beta(tau)}(s,a) ].

        La competencia puntua el MAXIMO de 5 episodios, no la media, asi que la
        politica optima para evaluar no tiene por que ser la neutral al riesgo.

        Medido sobre este juego, la intuicion falla: las distorsiones que buscan
        riesgo (`cvar_sup`, `wang`) empeoran tanto la media como la cola, y las
        que lo evitan (`cvar`, `pow`) mejoran ambas. Morir termina el episodio,
        asi que el unico camino a un score alto es sobrevivir; no hay ninguna
        apuesta que pague por arriesgarse.
        """
        modo, eta = self.distortion, self.distortion_eta
        if modo in (None, "neutral"):
            return taus
        if modo == "cvar":                    # aversion: solo la cola inferior
            return eta * taus
        if modo == "cvar_sup":                # busqueda: solo la cola superior
            return 1.0 - eta + eta * taus
        if modo == "wang":                    # eta > 0 desplaza tau hacia arriba
            normal = torch.distributions.Normal(
                torch.zeros((), device=taus.device), torch.ones((), device=taus.device)
            )
            return normal.cdf(normal.icdf(taus.clamp(1e-6, 1 - 1e-6)) + eta)
        if modo == "pow":                     # aversion suave: desplaza tau abajo
            return 1.0 - (1.0 - taus) ** (1.0 / (1.0 + abs(eta)))
        raise ValueError(f"distorsion desconocida: {modo}")

    def forward(self, obs: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Q(s, .) = E_tau[Z_beta(tau)(s, .)], estimado con k cuantiles."""
        k = k or self.k_eval
        taus = torch.rand(obs.shape[0], k, device=obs.device)
        return self.quantiles(obs, self.distorsionar(taus)).mean(dim=1)


def build_network(arch: str = "dqn", **kwargs) -> nn.Module:
    if arch == "dqn":
        kwargs.pop("n_cos", None)
        kwargs.pop("k_eval", None)
        return QNetwork(**kwargs)
    if arch == "iqn":
        return IQNNetwork(**kwargs)
    raise ValueError(f"arquitectura desconocida: {arch}")
