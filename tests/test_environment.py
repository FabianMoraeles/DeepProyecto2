"""Tests del entorno.

El test critico es test_rutas_equivalentes: garantiza que el preprocesamiento
de entrenamiento (AtariVectorEnv) y el de evaluacion (gym.make) producen
observaciones identicas. Si este test se rompe, cualquier score medido deja de
ser confiable.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from si_rl.envs import DEFAULT_CONFIG, describe_env, make_raw_env, make_single_env, make_vector_env

CFG_DETERMINISTA = dataclasses.replace(
    DEFAULT_CONFIG, repeat_action_probability=0.0, noop_max=0
)


def test_config_coincide_con_defaults_del_entorno():
    """EnvConfig debe replicar los defaults reales de ALE/SpaceInvaders-v5."""
    info = describe_env()
    kwargs = info["spec_kwargs"]
    assert kwargs["repeat_action_probability"] == DEFAULT_CONFIG.repeat_action_probability
    assert kwargs["frameskip"] == DEFAULT_CONFIG.frameskip
    assert kwargs["full_action_space"] == DEFAULT_CONFIG.full_action_space
    assert kwargs["max_num_frames_per_episode"] == DEFAULT_CONFIG.max_num_frames_per_episode


def test_sticky_actions_activas():
    """El default 0.25 de v5 debe estar activo: es lo que se evalua."""
    assert DEFAULT_CONFIG.repeat_action_probability == 0.25


def test_action_space():
    info = describe_env()
    assert info["action_meanings"] == [
        "NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"
    ]
    assert DEFAULT_CONFIG.n_actions == 6


def test_shape_observacion_preprocesada():
    env = make_single_env(DEFAULT_CONFIG)
    obs, _ = env.reset(seed=0)
    env.close()
    assert obs.shape == DEFAULT_CONFIG.obs_shape == (4, 84, 84)
    assert obs.dtype == np.uint8


def test_vector_env_shape():
    env = make_vector_env(CFG_DETERMINISTA, num_envs=4,
                          episodic_life=True, reward_clipping=True)
    obs, _ = env.reset(seed=0)
    env.close()
    assert obs.shape == (4, *DEFAULT_CONFIG.obs_shape)
    assert obs.dtype == np.uint8


@pytest.mark.parametrize("n_steps", [300])
def test_rutas_equivalentes(n_steps):
    """Vector env y gym.make deben dar observaciones y rewards identicos.

    Se desactivan sticky actions y noops para que ambos caminos sean
    deterministas y la comparacion sea significativa.
    """
    vec = make_vector_env(CFG_DETERMINISTA, num_envs=1,
                          episodic_life=False, reward_clipping=False)
    single = make_single_env(CFG_DETERMINISTA)

    obs_vec, _ = vec.reset(seed=123)
    obs_single, _ = single.reset(seed=123)
    assert np.array_equal(obs_vec[0], obs_single), "las observaciones de reset difieren"

    acciones = np.random.default_rng(0).integers(0, DEFAULT_CONFIG.n_actions, size=n_steps)
    score_vec = score_single = 0.0

    for i, a in enumerate(acciones):
        ov, rv, tv, uv, _ = vec.step(np.array([a]))
        os_, rs, ts, us, _ = single.step(int(a))
        score_vec += float(rv[0])
        score_single += float(rs)

        assert np.array_equal(ov[0], os_), f"las observaciones divergen en el step {i}"
        assert float(rv[0]) == float(rs), f"los rewards divergen en el step {i}"

        if bool(tv[0]) or ts:
            break

    vec.close()
    single.close()
    assert score_vec == score_single


def test_reward_sin_clipping_es_el_score_real():
    """Con reward_clipping=False los rewards son multiplos de la tabla de puntos."""
    env = make_vector_env(CFG_DETERMINISTA, num_envs=8,
                          episodic_life=False, reward_clipping=False)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    vistos = set()
    for _ in range(600):
        _, r, _, _, _ = env.step(rng.integers(0, 6, size=8))
        vistos.update(float(x) for x in r if x != 0)
    env.close()
    assert vistos, "no se observo ninguna recompensa positiva"
    assert all(v % 5 == 0 for v in vistos), f"recompensas inesperadas: {sorted(vistos)}"


def test_reward_clipping_acota_a_signo():
    env = make_vector_env(CFG_DETERMINISTA, num_envs=8,
                          episodic_life=False, reward_clipping=True)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(600):
        _, r, _, _, _ = env.step(rng.integers(0, 6, size=8))
        assert np.all(np.abs(r) <= 1.0)
    env.close()


def test_vidas_decrecen_y_episodio_termina():
    env = make_raw_env(DEFAULT_CONFIG)
    _, info = env.reset(seed=7)
    assert info["lives"] == 3
    vidas = [3]
    terminado = False
    for _ in range(5000):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        vidas.append(info["lives"])
        if term or trunc:
            terminado = True
            break
    env.close()
    assert terminado, "el episodio no termino en 5000 steps"
    assert min(vidas) == 0
    assert all(b <= a for a, b in zip(vidas, vidas[1:])), "las vidas no son monotonas"
