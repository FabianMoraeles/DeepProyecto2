from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from si_rl.baselines import FILA_JUGADOR, VERDE, _columnas_con_color
from si_rl.envs import DEFAULT_CONFIG, make_raw_env, make_single_env
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "reports" / "informe"


def posicion_jugador(pantalla: np.ndarray) -> int | None:
    cols = _columnas_con_color(pantalla, FILA_JUGADOR, VERDE)
    return int(cols.mean()) if cols.size else None


def senal_aleatoria(episodios: int) -> dict:
    env = make_raw_env(DEFAULT_CONFIG)
    rng = np.random.default_rng(0)
    valores, intervalos, pasos_totales, con_recompensa = Counter(), [], 0, 0
    duraciones, puntajes = [], []
    for ep in range(episodios):
        env.reset(seed=5000 + ep)
        ultimo, paso, score = 0, 0, 0.0
        while True:
            _, r, term, trunc, _ = env.step(int(rng.integers(6)))
            paso += 1
            if r != 0:
                valores[int(r)] += 1
                intervalos.append(paso - ultimo)
                ultimo = paso
                con_recompensa += 1
                score += r
            if term or trunc:
                break
        pasos_totales += paso
        duraciones.append(paso)
        puntajes.append(score)
    env.close()
    return {
        "episodios": episodios,
        "valores": dict(sorted(valores.items())),
        "intervalos": intervalos,
        "fraccion_pasos_con_recompensa": con_recompensa / pasos_totales,
        "pasos_totales": pasos_totales,
        "duraciones": duraciones,
        "puntajes": puntajes,
    }


def comportamiento_campeon(ruta: Path, episodios: int) -> dict:
    politica, env_cfg, meta = load_agent(ruta, epsilon=0.0)
    env = make_single_env(env_cfg)
    acciones, posiciones = Counter(), []
    finales, vidas_perdidas_en = [], []
    ultimos_ejemplo = None
    intervalos, pasos_totales, con_recompensa = [], 0, 0

    for ep in range(episodios):
        obs, info = env.reset(seed=200_000 + ep)
        vidas = info["lives"]
        score, paso, ultimo = 0.0, 0, 0
        cola = deque(maxlen=60)
        while True:
            a = politica(obs)
            obs, r, term, trunc, info = env.step(a)
            paso += 1
            acciones[a] += 1
            pantalla = env.unwrapped.ale.getScreenRGB()
            cola.append(pantalla.copy())
            x = posicion_jugador(pantalla)
            if x is not None:
                posiciones.append(x)
            if r != 0:
                score += r
                intervalos.append(paso - ultimo)
                ultimo = paso
                con_recompensa += 1
            if info["lives"] < vidas:
                vidas_perdidas_en.append({"episodio": ep, "puntaje": score, "paso": paso,
                                          "vidas_restantes": int(info["lives"])})
                vidas = info["lives"]
            if term or trunc:
                break
        pasos_totales += paso
        finales.append({"episodio": ep, "puntaje": score, "pasos": paso,
                        "vidas_al_terminar": int(info["lives"]),
                        "terminated": bool(term), "truncated": bool(trunc)})
        if ultimos_ejemplo is None and abs(score - 2815) < 1:
            ultimos_ejemplo = np.stack(list(cola))
        print(f"  ep {ep:3d}  puntaje={score:7.1f}  pasos={paso:5d}  "
              f"vidas al terminar={info['lives']}", flush=True)
    env.close()

    if ultimos_ejemplo is not None:
        np.save(SALIDA / "ultimos_fotogramas_2815.npy", ultimos_ejemplo)

    return {
        "checkpoint": str(ruta),
        "step": meta["step"],
        "acciones": {int(k): v for k, v in sorted(acciones.items())},
        "posiciones": posiciones,
        "finales": finales,
        "vidas_perdidas": vidas_perdidas_en,
        "intervalos": intervalos,
        "fraccion_pasos_con_recompensa": con_recompensa / pasos_totales,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios-aleatorio", type=int, default=40)
    parser.add_argument("--episodios-campeon", type=int, default=40)
    parser.add_argument("--checkpoint", default=str(RAIZ / "checkpoints" / "champion.pt"))
    args = parser.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)

    print("Senal de recompensa, politica aleatoria")
    aleatorio = senal_aleatoria(args.episodios_aleatorio)
    (SALIDA / "senal_aleatoria.json").write_text(json.dumps(aleatorio), encoding="utf-8")
    print(f"  valores {aleatorio['valores']}")
    print(f"  fraccion de pasos con recompensa {aleatorio['fraccion_pasos_con_recompensa']:.4f}")
    print(f"  intervalo medio {np.mean(aleatorio['intervalos']):.1f} pasos")

    print("\nComportamiento del campeon")
    campeon = comportamiento_campeon(Path(args.checkpoint), args.episodios_campeon)
    (SALIDA / "comportamiento_campeon.json").write_text(json.dumps(campeon), encoding="utf-8")

    fin = campeon["finales"]
    print(f"\n  acciones {campeon['acciones']}")
    print(f"  vidas al terminar {Counter(f['vidas_al_terminar'] for f in fin)}")
    print(f"  terminated {sum(f['terminated'] for f in fin)}  truncated {sum(f['truncated'] for f in fin)}")
    print(f"  fraccion de pasos con recompensa {campeon['fraccion_pasos_con_recompensa']:.4f}")
    print(f"  intervalo medio {np.mean(campeon['intervalos']):.1f} pasos")
    perdidas = campeon["vidas_perdidas"]
    for v in (2, 1, 0):
        sub = [p["puntaje"] for p in perdidas if p["vidas_restantes"] == v]
        if sub:
            print(f"  puntaje al quedar con {v} vidas: media {np.mean(sub):.0f}  "
                  f"mediana {np.median(sub):.0f}  n={len(sub)}")


if __name__ == "__main__":
    main()
