"""Tests del reescalado invertible de valores."""

from __future__ import annotations

import pytest
import torch

from si_rl.agents import h, h_inv


def test_h_es_invertible():
    """h_inv(h(x)) == x en todo el rango de retornos que puede ver el agente."""
    x = torch.tensor([-5000.0, -200.0, -30.0, -1.0, 0.0, 1.0, 5.0, 30.0,
                      200.0, 2815.0, 50_000.0])
    assert torch.allclose(h_inv(h(x)), x, atol=1e-2, rtol=1e-4)


def test_h_preserva_el_orden():
    """Si no preservara el orden, cambiaria que accion es la mejor."""
    x = torch.linspace(-3000, 3000, 2000)
    y = h(x)
    assert torch.all(y[1:] > y[:-1])


def test_h_comprime_el_rango():
    """El punto es que Q quepa en un rango manejable sin perder el orden."""
    assert h(torch.tensor(200.0)) < 20.0
    assert h(torch.tensor(5000.0)) < 130.0


def test_h_distingue_recompensas_que_el_clipping_iguala():
    """Clipar hace que un alien de 5 y la nodriza de 200 sean identicos."""
    alien_debil, alien_fuerte, nodriza = 5.0, 30.0, 200.0
    valores = h(torch.tensor([alien_debil, alien_fuerte, nodriza]))
    assert valores[0] < valores[1] < valores[2]
    # con clipping los tres colapsan al mismo valor
    clipeados = torch.sign(torch.tensor([alien_debil, alien_fuerte, nodriza]))
    assert clipeados[0] == clipeados[1] == clipeados[2]


def test_h_es_impar():
    x = torch.tensor([0.5, 7.0, 150.0])
    assert torch.allclose(h(-x), -h(x), atol=1e-6)


@pytest.mark.parametrize("valor", [0.0, 1.0, -1.0, 630.0, -630.0])
def test_h_inv_puntual(valor):
    t = torch.tensor(valor)
    assert h_inv(h(t)).item() == pytest.approx(valor, abs=1e-3)
