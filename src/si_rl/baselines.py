"""Politicas de referencia sin aprendizaje.

Sirven como piso del informe: cualquier agente entrenado debe superarlas de
forma clara. El agente de regla simple viene del Laboratorio 5 y opera sobre
pixeles RGB crudos, no sobre la observacion preprocesada.
"""

from __future__ import annotations

import numpy as np

# Colores de referencia (R, G, B) y bandas de filas del agente de regla simple.
VERDE = (50, 132, 50)
AMARILLO = (134, 134, 29)
GRIS = (142, 142, 142)
FILA_JUGADOR = (180, 205)
FILA_INVASORES = (20, 172)
FILA_PROYECTILES = (20, 180)
TOLERANCIA = 4
MARGEN_ESQUIVE = 8
BORDE_IZQ = 25
BORDE_DER = 135
MARGEN_PROPIO = 2
DURACION_DISPARO = 18


class RandomPolicy:
    """Accion uniforme sobre el espacio de acciones."""

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(self.n_actions))

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.integers(self.n_actions, size=len(obs))


def _columnas_con_color(img: np.ndarray, filas: tuple[int, int],
                        color: tuple[int, int, int]) -> np.ndarray:
    """Columnas de la franja de filas dada que contienen el color exacto."""
    r0, r1 = filas
    franja = img[r0:r1]
    mascara = (
        (franja[:, :, 0] == color[0])
        & (franja[:, :, 1] == color[1])
        & (franja[:, :, 2] == color[2])
    )
    return np.where(mascara.any(axis=0))[0]


class RuleBasedPolicy:
    """Agente reactivo por color de pixel (portado del Lab 5).

    Localiza al jugador y al invasor mas cercano, esquiva proyectiles enemigos
    y en caso contrario avanza disparando. Distingue su propio disparo en vuelo
    de las bombas enemigas mediante un contador de frames.
    """

    def __init__(self, action_meanings: list[str]):
        def idx(nombre: str, alt: int = 0) -> int:
            return action_meanings.index(nombre) if nombre in action_meanings else alt

        self.fuego = idx("FIRE")
        self.derecha = idx("RIGHT", self.fuego)
        self.izquierda = idx("LEFT", self.fuego)
        self.fuego_der = idx("RIGHTFIRE", self.derecha)
        self.fuego_izq = idx("LEFTFIRE", self.izquierda)
        self.acciones_que_disparan = {self.fuego, self.fuego_der, self.fuego_izq}
        self.reset()

    def reset(self) -> None:
        self._cooldown = 0
        self._x_disparo: int | None = None

    def __call__(self, obs: np.ndarray) -> int:
        columnas_jugador = _columnas_con_color(obs, FILA_JUGADOR, VERDE)
        if columnas_jugador.size == 0:
            return self.fuego
        x = int(columnas_jugador.mean())

        if self._cooldown > 0:
            self._cooldown -= 1

        columnas_proyectiles = _columnas_con_color(obs, FILA_PROYECTILES, GRIS)
        amenazas = [
            int(c)
            for c in columnas_proyectiles
            if not (
                self._cooldown > 0
                and self._x_disparo is not None
                and abs(int(c) - self._x_disparo) <= MARGEN_PROPIO
            )
        ]

        accion = None
        if amenazas:
            arr = np.array(amenazas)
            distancias = np.abs(arr - x)
            if distancias.min() <= MARGEN_ESQUIVE:
                cercana = int(arr[np.argmin(distancias)])
                if x <= BORDE_IZQ:
                    accion = self.fuego_der
                elif x >= BORDE_DER:
                    accion = self.fuego_izq
                elif cercana >= x:
                    accion = self.fuego_izq
                else:
                    accion = self.fuego_der

        if accion is None:
            columnas_invasores = _columnas_con_color(obs, FILA_INVASORES, AMARILLO)
            if columnas_invasores.size == 0:
                accion = self.fuego
            else:
                distancias = np.abs(columnas_invasores - x)
                objetivo = int(columnas_invasores[np.argmin(distancias)])
                delta = objetivo - x
                if abs(delta) <= TOLERANCIA:
                    accion = self.fuego
                elif delta > 0:
                    accion = self.fuego_der
                else:
                    accion = self.fuego_izq

        if accion in self.acciones_que_disparan and self._cooldown <= 0:
            self._cooldown = DURACION_DISPARO
            self._x_disparo = x

        return accion
