"""Harness de evaluacion.

Separado del entrenamiento a proposito: aqui nunca se actualizan pesos ni se
escribe al replay buffer. El score reportado es siempre el score real del juego
(sin clipping, sin episodic-life, sin reward shaping).

La metrica de la competencia es el MAXIMO de 5 episodios, no la media. Por eso
ademas de las estadisticas habituales se estima E[max de 5] por bootstrap, que
es lo que realmente predice el desempeno el dia de la presentacion.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

Policy = Callable[[np.ndarray], int]

COMPETITION_EPISODES = 5


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    score: float
    steps: int
    lives_lost: int
    terminated: bool
    truncated: bool
    duration_s: float


@dataclass
class EvalReport:
    label: str
    n_episodes: int
    scores: list[float]
    episodes: list[EpisodeResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def median(self) -> float:
        return float(np.median(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores))

    @property
    def max_score(self) -> float:
        return float(np.max(self.scores))

    @property
    def min_score(self) -> float:
        return float(np.min(self.scores))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.scores, q))

    def expected_best_of(self, k: int = COMPETITION_EPISODES, n_boot: int = 20_000,
                         seed: int = 0) -> dict:
        """Estima la distribucion del maximo de k episodios por bootstrap.

        Esta es la metrica de la competencia. Con pocos episodios evaluados la
        estimacion es ruidosa, pero sigue siendo mejor guia que la media.
        """
        rng = np.random.default_rng(seed)
        draws = rng.choice(self.scores, size=(n_boot, k), replace=True).max(axis=1)
        return {
            "k": k,
            "mean": float(draws.mean()),
            "p10": float(np.percentile(draws, 10)),
            "p50": float(np.percentile(draws, 50)),
            "p90": float(np.percentile(draws, 90)),
        }

    def summary(self) -> dict:
        best_of = self.expected_best_of()
        return {
            "label": self.label,
            "n_episodes": self.n_episodes,
            "mean": round(self.mean, 1),
            "median": round(self.median, 1),
            "std": round(self.std, 1),
            "min": self.min_score,
            "max": self.max_score,
            "p90": round(self.percentile(90), 1),
            f"expected_best_of_{COMPETITION_EPISODES}": round(best_of["mean"], 1),
            "metadata": self.metadata,
        }

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "scores": self.scores,
            "episodes": [asdict(e) for e in self.episodes],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def __str__(self) -> str:
        s = self.summary()
        return (
            f"{self.label:28s} n={s['n_episodes']:4d}  "
            f"mean={s['mean']:8.1f}  median={s['median']:8.1f}  "
            f"std={s['std']:7.1f}  max={s['max']:8.1f}  "
            f"E[best-of-5]={s[f'expected_best_of_{COMPETITION_EPISODES}']:8.1f}"
        )


def run_episode(env, policy: Policy, seed: int | None = None,
                max_steps: int = 30_000, episode: int = 0) -> EpisodeResult:
    """Ejecuta un episodio completo y devuelve el score real acumulado."""
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)
    lives = info.get("lives", 0)
    lives_lost = 0
    score = 0.0
    steps = 0
    terminated = truncated = False
    t0 = time.perf_counter()

    while steps < max_steps:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += float(reward)
        steps += 1

        current_lives = info.get("lives", lives)
        if current_lives < lives:
            lives_lost += 1
        lives = current_lives

        if terminated or truncated:
            break

    return EpisodeResult(
        episode=episode,
        seed=seed if seed is not None else -1,
        score=score,
        steps=steps,
        lives_lost=lives_lost,
        terminated=terminated,
        truncated=truncated,
        duration_s=time.perf_counter() - t0,
    )


def evaluate(make_env_fn: Callable[[], object], policy: Policy, *,
             label: str, n_episodes: int = COMPETITION_EPISODES,
             base_seed: int = 1000, max_steps: int = 30_000,
             metadata: dict | None = None, verbose: bool = True) -> EvalReport:
    """Evalua una politica en n episodios independientes con seeds deterministas."""
    env = make_env_fn()
    results: list[EpisodeResult] = []
    try:
        for i in range(n_episodes):
            result = run_episode(env, policy, seed=base_seed + i,
                                 max_steps=max_steps, episode=i)
            results.append(result)
            if verbose:
                print(f"  [{label}] ep {i + 1:3d}/{n_episodes}  "
                      f"score={result.score:8.1f}  steps={result.steps:6d}  "
                      f"{result.duration_s:5.2f}s")
    finally:
        env.close()

    return EvalReport(
        label=label,
        n_episodes=n_episodes,
        scores=[r.score for r in results],
        episodes=results,
        metadata=metadata or {},
    )


def evaluate_vectorized(env_cfg, policy, *, label: str, n_episodes: int = 100,
                        num_envs: int = 32, base_seed: int = 1000,
                        max_env_steps: int = 200_000,
                        metadata: dict | None = None) -> EvalReport:
    """Evalua en paralelo sobre el entorno vectorizado.

    Permite 100 episodios en el tiempo que el camino de un solo entorno tarda en
    hacer 10, lo que importa porque con 10 episodios la desviacion del score es
    tan grande que elegir checkpoint por la media selecciona suerte, no modelo.

    Cada entorno corre una cuota fija de episodios. Sin esa cuota se contarian
    los primeros N episodios en terminar, que son sistematicamente los mas
    cortos, y el score quedaria sesgado hacia abajo.

    `policy` debe exponer act_batch((N, C, H, W)) -> (N,).
    """
    from si_rl.envs import make_vector_env

    env = make_vector_env(env_cfg, num_envs=num_envs,
                          episodic_life=False, reward_clipping=False)
    cuota = -(-n_episodes // num_envs)
    completados = np.zeros(num_envs, dtype=int)
    puntajes: list[float] = []
    resultados: list[EpisodeResult] = []
    score_actual = np.zeros(num_envs)
    pasos_actual = np.zeros(num_envs, dtype=int)
    prev_fin = np.zeros(num_envs, dtype=bool)
    t0 = time.perf_counter()

    obs, _ = env.reset(seed=base_seed)
    pasos = 0
    try:
        while np.any(completados < cuota) and pasos < max_env_steps:
            acciones = policy.act_batch(obs)
            obs, recompensas, term, trunc, _ = env.step(acciones)
            fin = term | trunc
            activo = ~prev_fin

            score_actual += np.where(activo, recompensas, 0.0)
            pasos_actual += activo.astype(int)

            for e in np.flatnonzero(fin):
                if completados[e] < cuota:
                    puntajes.append(float(score_actual[e]))
                    resultados.append(EpisodeResult(
                        episode=len(resultados), seed=base_seed,
                        score=float(score_actual[e]), steps=int(pasos_actual[e]),
                        lives_lost=3, terminated=bool(term[e]), truncated=bool(trunc[e]),
                        duration_s=0.0,
                    ))
                    completados[e] += 1
                score_actual[e] = 0.0
                pasos_actual[e] = 0

            prev_fin = fin
            pasos += 1
    finally:
        env.close()

    meta = dict(metadata or {})
    meta.update({"num_envs": num_envs, "duracion_s": round(time.perf_counter() - t0, 1)})
    return EvalReport(label=label, n_episodes=len(puntajes), scores=puntajes,
                      episodes=resultados, metadata=meta)


def record_episode(make_env_fn: Callable[[], object], policy: Policy, *,
                   seed: int | None = None, max_steps: int = 30_000) -> EpisodeResult:
    """Ejecuta un episodio con un entorno que ya trae RecordVideo."""
    env = make_env_fn()
    try:
        return run_episode(env, policy, seed=seed, max_steps=max_steps)
    finally:
        env.close()
