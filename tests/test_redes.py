"""Tests de arquitecturas y de ida y vuelta de checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from si_rl.envs import DEFAULT_CONFIG
from si_rl.networks import IQNNetwork, QNetwork, build_network
from si_rl.policies import load_agent, save_checkpoint

B, A = 5, 6


def test_qnetwork_shape():
    net = QNetwork(n_actions=A)
    obs = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8)
    assert net(obs).shape == (B, A)


def test_dueling_cancela_la_media_de_la_ventaja():
    """Q - V debe tener media cero sobre las acciones, por construccion."""
    net = QNetwork(n_actions=A, dueling=True)
    obs = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        z = net.encoder(obs.float() / 255.0)
        v = net.valor(z)
        q = net(obs)
    assert torch.allclose((q - v).mean(dim=1), torch.zeros(B), atol=1e-5)


@pytest.mark.parametrize("dueling", [False, True])
def test_iqn_shapes(dueling):
    net = IQNNetwork(n_actions=A, dueling=dueling)
    obs = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8)
    taus = torch.rand(B, 8)
    assert net.quantiles(obs, taus).shape == (B, 8, A)
    assert net(obs).shape == (B, A)


def test_iqn_los_cuantiles_dependen_de_tau():
    """Si la salida no dependiera de tau, la red seria un DQN caro."""
    net = IQNNetwork(n_actions=A)
    obs = torch.randint(0, 255, (1, 4, 84, 84), dtype=torch.uint8)
    taus = torch.tensor([[0.05, 0.95]])
    with torch.no_grad():
        z = net.quantiles(obs, taus)
    assert not torch.allclose(z[0, 0], z[0, 1], atol=1e-4)


@pytest.mark.parametrize("modo,eta", [("neutral", 0.0), ("cvar_sup", 0.25),
                                      ("cvar", 0.5), ("wang", 0.75), ("pow", 1.0)])
def test_distorsiones_de_riesgo_mapean_dentro_de_cero_uno(modo, eta):
    net = IQNNetwork(n_actions=A)
    net.distortion, net.distortion_eta = modo, eta
    taus = torch.rand(200)
    out = net.distorsionar(taus)
    assert torch.all((out >= 0) & (out <= 1)), f"{modo} salio de [0,1]"


def test_cvar_sup_mira_la_cola_superior_y_cvar_la_inferior():
    """La distorsion debe mover la masa hacia el extremo correcto."""
    net = IQNNetwork(n_actions=A)
    taus = torch.rand(5000)

    net.distortion, net.distortion_eta = "cvar_sup", 0.25
    assert net.distorsionar(taus).min() >= 0.75 - 1e-6

    net.distortion, net.distortion_eta = "cvar", 0.25
    assert net.distorsionar(taus).max() <= 0.25 + 1e-6

    net.distortion = "wang"
    net.distortion_eta = 0.75
    assert net.distorsionar(taus).mean() > taus.mean()

    # pow con eta>0 desplaza tau hacia abajo: es aversion al riesgo, no busqueda
    net.distortion, net.distortion_eta = "pow", 1.0
    assert net.distorsionar(taus).mean() < taus.mean()


def test_distorsion_desconocida_falla():
    net = IQNNetwork(n_actions=A)
    net.distortion = "inventada"
    with pytest.raises(ValueError):
        net.distorsionar(torch.rand(4))


def test_build_network_dispatch():
    assert isinstance(build_network("dqn", n_actions=A), QNetwork)
    assert isinstance(build_network("iqn", n_actions=A), IQNNetwork)
    with pytest.raises(ValueError):
        build_network("rainbow", n_actions=A)


@pytest.mark.parametrize("arch,kwargs", [
    ("dqn", {"dueling": False}),
    ("dqn", {"dueling": True}),
    ("iqn", {"dueling": True, "n_cos": 32, "k_eval": 16}),
])
def test_checkpoint_ida_y_vuelta(tmp_path, arch, kwargs):
    """Cargar un checkpoint debe reproducir exactamente la misma inferencia.

    Es el requisito del dia de la presentacion: los pesos y la arquitectura
    tienen que viajar juntos.
    """
    net_kwargs = dict(in_channels=4, n_actions=A, **kwargs)
    net = build_network(arch, **net_kwargs)
    ruta = tmp_path / "ckpt.pt"
    save_checkpoint(ruta, net=net, env_config=DEFAULT_CONFIG, train_config={},
                    step=123, arch=arch, net_kwargs=net_kwargs)

    politica, env_cfg, meta = load_agent(ruta, device="cpu")
    assert meta["step"] == 123
    assert env_cfg == DEFAULT_CONFIG

    obs = torch.randint(0, 255, (B, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        if arch == "iqn":
            taus = torch.rand(B, 8)
            original = net.quantiles(obs, taus)
            cargado = politica.net.quantiles(obs, taus)
        else:
            original = net(obs)
            cargado = politica.net(obs)
    assert torch.allclose(original, cargado, atol=1e-6)


def test_politica_greedy_funciona_con_iqn():
    net = build_network("iqn", in_channels=4, n_actions=A)
    from si_rl.policies import GreedyPolicy
    politica = GreedyPolicy(net, torch.device("cpu"))
    obs = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    assert 0 <= politica(obs) < A
    lote = np.random.randint(0, 255, (7, 4, 84, 84), dtype=np.uint8)
    assert politica.act_batch(lote).shape == (7,)
