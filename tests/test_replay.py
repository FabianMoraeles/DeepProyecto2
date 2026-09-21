"""Tests del replay buffer.

El test decisivo es test_stack_reconstruido_coincide_con_el_entorno: comprueba
contra el entorno real que la deduplicacion de frames reconstruye exactamente
la observacion que vio el agente. Un error silencioso aqui envenena todo el
entrenamiento sin que ninguna metrica lo delate.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from si_rl.envs import DEFAULT_CONFIG, make_vector_env
from si_rl.replay import ReplayBuffer

CFG = dataclasses.replace(DEFAULT_CONFIG, repeat_action_probability=0.0)


def test_stack_reconstruido_coincide_con_el_entorno():
    num_envs = 4
    env = make_vector_env(CFG, num_envs=num_envs, episodic_life=False,
                          reward_clipping=False)
    buf = ReplayBuffer(capacity=num_envs * 600, num_envs=num_envs, n_step=1)

    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    esperados: dict[tuple[int, int], np.ndarray] = {}
    prev_term = np.zeros(num_envs, dtype=bool)

    for _ in range(500):
        acciones = rng.integers(0, CFG.n_actions, size=num_envs)
        next_obs, r, term, trunc, _ = env.step(acciones)
        fin = term | trunc
        dummy = prev_term

        frame = np.where(dummy[:, None, None], next_obs[:, -1], obs[:, -1])
        slot = buf.pos
        buf.add(frame, acciones, r, fin, dummy=dummy)

        for e in range(num_envs):
            if not dummy[e]:
                esperados[(slot, e)] = obs[e].copy()

        prev_term = fin
        obs = next_obs

    env.close()

    assert len(esperados) > 1500
    slots = np.array([s for s, _ in esperados])
    envs = np.array([e for _, e in esperados])
    reconstruido = buf._build_stack(slots, envs)
    referencia = np.stack(list(esperados.values()))

    assert reconstruido.shape == referencia.shape
    assert np.array_equal(reconstruido, referencia), (
        "la reconstruccion del stack no coincide con la observacion real"
    )


def test_padding_con_ceros_al_inicio_del_episodio():
    """Los primeros pasos de un episodio se rellenan con ceros, como hace ALE."""
    buf = ReplayBuffer(capacity=64, num_envs=1, n_step=1)
    for paso in range(4):
        frame = np.full((1, 84, 84), paso + 1, dtype=np.uint8)
        buf.add(frame, np.array([0]), np.array([0.0]), np.array([False]))

    stack0 = buf._build_stack(np.array([0]), np.array([0]))[0]
    assert np.array_equal(stack0[:3], np.zeros((3, 84, 84), dtype=np.uint8))
    assert stack0[3][0, 0] == 1

    stack3 = buf._build_stack(np.array([3]), np.array([0]))[0]
    assert [int(stack3[k][0, 0]) for k in range(4)] == [1, 2, 3, 4]


def test_retorno_n_step():
    gamma = 0.5
    buf = ReplayBuffer(capacity=128, num_envs=1, gamma=gamma, n_step=3)
    recompensas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    for i, r in enumerate(recompensas):
        buf.add(np.full((1, 84, 84), i, dtype=np.uint8), np.array([0]),
                np.array([r]), np.array([False]))

    lote = buf.sample(64)
    for fila in range(64):
        s = int(lote.indices[fila, 0])
        esperado = sum(gamma**k * recompensas[s + k] for k in range(3))
        assert lote.returns[fila] == pytest.approx(esperado, rel=1e-5)
        assert lote.n_steps[fila] == 3
        assert not lote.done[fila]


def test_n_step_se_corta_en_la_terminacion():
    gamma = 0.9
    buf = ReplayBuffer(capacity=128, num_envs=1, gamma=gamma, n_step=3)
    recompensas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    terminales = [False, True, False, False, False, False]
    for i, (r, t) in enumerate(zip(recompensas, terminales)):
        buf.add(np.full((1, 84, 84), i, dtype=np.uint8), np.array([0]),
                np.array([r]), np.array([t]))

    lote = buf.sample(256)
    desde_cero = lote.indices[:, 0] == 0
    assert desde_cero.any()
    # ventana que arranca en 0: 1.0 + 0.9*2.0, y se corta por la terminacion
    assert lote.returns[desde_cero][0] == pytest.approx(1.0 + gamma * 2.0, rel=1e-5)
    assert lote.n_steps[desde_cero][0] == 2
    assert lote.done[desde_cero][0]


def test_los_slots_dummy_nunca_se_muestrean():
    buf = ReplayBuffer(capacity=256, num_envs=2, n_step=1)
    rng = np.random.default_rng(0)
    esperado_invalido = set()

    prev_term = np.zeros(2, dtype=bool)
    for paso in range(100):
        term = np.array([paso % 17 == 16, paso % 23 == 22])
        slot = buf.pos
        buf.add(rng.integers(0, 255, size=(2, 84, 84), dtype=np.uint8),
                rng.integers(0, 6, size=2), rng.random(2).astype(np.float32),
                term, dummy=prev_term)
        for e in range(2):
            if prev_term[e]:
                esperado_invalido.add((slot, e))
        prev_term = term

    assert esperado_invalido
    lote = buf.sample(4000)
    muestreados = {(int(s), int(e)) for s, e in lote.indices}
    assert not (muestreados & esperado_invalido)


def test_prefetch_entrega_lotes_bajo_escritura_concurrente():
    """El muestreo en segundo plano debe seguir siendo valido mientras se escribe."""
    from si_rl.replay import PrefetchSampler

    num_envs = 8
    buf = ReplayBuffer(capacity=num_envs * 300, num_envs=num_envs, n_step=3)
    rng = np.random.default_rng(0)
    for _ in range(80):
        buf.add(rng.integers(0, 255, size=(num_envs, 84, 84), dtype=np.uint8),
                rng.integers(0, 6, size=num_envs), rng.random(num_envs).astype(np.float32),
                np.zeros(num_envs, dtype=bool))

    sampler = PrefetchSampler(buf, batch_size=64, profundidad=2)
    try:
        for _ in range(200):
            buf.add(rng.integers(0, 255, size=(num_envs, 84, 84), dtype=np.uint8),
                    rng.integers(0, 6, size=num_envs),
                    rng.random(num_envs).astype(np.float32),
                    rng.random(num_envs) < 0.01)
            lote = sampler.get()
            assert lote.obs.shape == (64, 4, 84, 84)
            assert lote.next_obs.shape == (64, 4, 84, 84)
            assert lote.actions.shape == (64,)
            assert np.all(lote.n_steps >= 1) and np.all(lote.n_steps <= 3)
    finally:
        sampler.close()


def test_memoria_deduplicada():
    """El buffer debe pesar ~1 frame por transicion, no 4."""
    buf = ReplayBuffer(capacity=10_000, num_envs=10, n_step=1)
    bytes_por_transicion = buf.nbytes / buf.capacity
    assert bytes_por_transicion < 1.2 * 84 * 84
