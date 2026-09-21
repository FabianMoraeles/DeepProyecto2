"""Tests del arbol de sumas y del replay priorizado."""

from __future__ import annotations

import numpy as np
import pytest

from si_rl.replay import PrioritizedReplayBuffer, SumTree


def test_sumtree_total_y_lectura():
    tree = SumTree(8)
    tree.set(np.array([0, 3, 7]), np.array([1.0, 2.0, 5.0]))
    assert tree.total == pytest.approx(8.0)
    tree.set(np.array([3]), np.array([10.0]))
    assert tree.total == pytest.approx(16.0)


def test_sumtree_maneja_indices_repetidos():
    """Con indices repetidos el delta se calcularia sobre un valor ya pisado."""
    tree = SumTree(8)
    tree.set(np.array([2, 2, 2]), np.array([1.0, 2.0, 3.0]))
    assert tree.total == pytest.approx(1.0)


def test_sumtree_muestrea_proporcional_a_la_prioridad():
    tree = SumTree(4)
    tree.set(np.arange(4), np.array([1.0, 0.0, 3.0, 0.0]))
    rng = np.random.default_rng(0)
    hojas, prob = tree.sample(4000, rng)

    assert set(np.unique(hojas)) <= {0, 2}
    frac_0 = float((hojas == 0).mean())
    assert frac_0 == pytest.approx(0.25, abs=0.02)
    assert prob[hojas == 0] == pytest.approx(0.25, abs=1e-6)
    assert prob[hojas == 2] == pytest.approx(0.75, abs=1e-6)


def test_sumtree_prioridad_cero_nunca_se_muestrea():
    tree = SumTree(16)
    valores = np.zeros(16)
    valores[5] = 1.0
    tree.set(np.arange(16), valores)
    hojas, _ = tree.sample(500, np.random.default_rng(1))
    assert np.all(hojas == 5)


def _llenar(buf, pasos, rng, p_terminal=0.0):
    for _ in range(pasos):
        buf.add(rng.integers(0, 255, size=(buf.num_envs, 84, 84), dtype=np.uint8),
                rng.integers(0, 6, size=buf.num_envs),
                rng.random(buf.num_envs).astype(np.float32),
                rng.random(buf.num_envs) < p_terminal)


def test_per_no_muestrea_ventanas_n_step_incompletas():
    """Un slot solo es muestreable cuando sus n sucesores ya se escribieron."""
    num_envs = 4
    buf = PrioritizedReplayBuffer(capacity=num_envs * 200, num_envs=num_envs, n_step=3)
    rng = np.random.default_rng(0)
    _llenar(buf, 60, rng)

    lote = buf.sample(500)
    slots = lote.indices[:, 0]
    # los ultimos n_step slots escritos no pueden aparecer
    recientes = {(buf.pos - 1 - k) % buf.num_slots for k in range(buf.n_step)}
    assert not (set(np.unique(slots)) & recientes)


def test_per_pesos_de_importancia_normalizados():
    num_envs = 4
    buf = PrioritizedReplayBuffer(capacity=num_envs * 200, num_envs=num_envs, n_step=1)
    rng = np.random.default_rng(0)
    _llenar(buf, 80, rng)

    lote = buf.sample(128)
    assert lote.weights is not None
    assert lote.weights.max() == pytest.approx(1.0)
    assert np.all(lote.weights > 0)


def test_per_prioriza_las_transiciones_con_mayor_error():
    num_envs = 4
    buf = PrioritizedReplayBuffer(capacity=num_envs * 200, num_envs=num_envs,
                                  n_step=1, alpha=1.0)
    rng = np.random.default_rng(0)
    _llenar(buf, 80, rng)

    lote = buf.sample(256)
    # se le da error alto solo a la primera transicion del lote
    errores = np.full(256, 0.001)
    errores[0] = 100.0
    destacada = tuple(lote.indices[0])
    buf.update_priorities(lote.indices, errores)

    nuevo = buf.sample(400)
    veces = sum(1 for fila in nuevo.indices if tuple(fila) == destacada)
    assert veces > 40, f"la transicion priorizada solo salio {veces} veces"


def test_per_beta_se_recoce_hasta_uno():
    buf = PrioritizedReplayBuffer(capacity=64, num_envs=4, n_step=1,
                                  beta_start=0.4, beta_end=1.0, beta_steps=1000)
    buf.set_beta_por_paso(0)
    assert buf.beta == pytest.approx(0.4)
    buf.set_beta_por_paso(500)
    assert buf.beta == pytest.approx(0.7)
    buf.set_beta_por_paso(5000)
    assert buf.beta == pytest.approx(1.0)
