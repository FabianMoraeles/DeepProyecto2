"""Tests del harness de evaluacion."""

from __future__ import annotations

import numpy as np

from si_rl.baselines import RandomPolicy
from si_rl.envs import DEFAULT_CONFIG, make_single_env
from si_rl.evaluate import EvalReport, evaluate, evaluate_vectorized


def test_expected_best_of_usa_el_maximo():
    """E[max de k] debe quedar entre la media y el maximo observado."""
    reporte = EvalReport(label="t", n_episodes=6,
                         scores=[100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    est = reporte.expected_best_of(k=5)
    assert reporte.mean < est["mean"] <= reporte.max_score


def test_expected_best_of_con_scores_constantes():
    reporte = EvalReport(label="t", n_episodes=4, scores=[250.0] * 4)
    assert reporte.expected_best_of(k=5)["mean"] == 250.0


def test_evaluacion_vectorizada_respeta_la_cuota_por_entorno():
    """Sin cuota se contarian los episodios mas cortos primero y el score se sesga."""
    politica = RandomPolicy(DEFAULT_CONFIG.n_actions, seed=0)
    reporte = evaluate_vectorized(DEFAULT_CONFIG, politica, label="vec",
                                  n_episodes=32, num_envs=8, base_seed=99)
    assert reporte.n_episodes == 32
    assert all(s > 0 for s in reporte.scores)


def test_rutas_de_evaluacion_dan_la_misma_distribucion():
    """La ruta vectorizada y la de un solo entorno miden lo mismo.

    Ambas comparten preprocesamiento bit-identico, asi que con la misma politica
    las medias solo pueden diferir por ruido de muestreo.
    """
    n = 40
    unica = evaluate(lambda: make_single_env(DEFAULT_CONFIG),
                     RandomPolicy(DEFAULT_CONFIG.n_actions, seed=1),
                     label="unica", n_episodes=n, base_seed=3000, verbose=False)
    vect = evaluate_vectorized(DEFAULT_CONFIG,
                               RandomPolicy(DEFAULT_CONFIG.n_actions, seed=1),
                               label="vect", n_episodes=n, num_envs=8, base_seed=3000)

    # error estandar de la diferencia de medias, con margen de 4 sigma
    se = np.sqrt(unica.std**2 / n + vect.std**2 / n)
    assert abs(unica.mean - vect.mean) < 4 * se, (
        f"medias incompatibles: {unica.mean:.1f} vs {vect.mean:.1f} (se={se:.1f})"
    )
