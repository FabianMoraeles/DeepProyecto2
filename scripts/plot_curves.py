from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "runs"
FIGURAS = RAIZ / "reports" / "figures"

PANELES = [
    ("eval/score_medio", "Score de evaluacion (politica greedy)", "score"),
    ("episodio/score_medio_200", "Score de entrenamiento (media de 200 ep)", "score"),
    ("entrenamiento/q_medio", "Q medio", "Q"),
    ("entrenamiento/perdida", "Perdida", "perdida"),
]


def serie(acc, tag):
    if tag not in acc.Tags()["scalars"]:
        return [], []
    puntos = acc.Scalars(tag)
    return [p.step for p in puntos], [p.value for p in puntos]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--salida", default="curvas_entrenamiento.png")
    args = parser.parse_args()

    nombres = args.runs or sorted(
        d.name for d in RUNS.iterdir() if (d / "tensorboard").exists()
    )
    if not nombres:
        raise SystemExit("no hay corridas con datos de tensorboard")

    accs = {}
    for nombre in nombres:
        acc = EventAccumulator(str(RUNS / nombre / "tensorboard"),
                               size_guidance={"scalars": 100_000})
        acc.Reload()
        accs[nombre] = acc

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (tag, titulo, ylabel) in zip(axes.flat, PANELES):
        for nombre, acc in accs.items():
            x, y = serie(acc, tag)
            if x:
                ax.plot(x, y, label=nombre, linewidth=1.4)
        ax.set_title(titulo)
        ax.set_xlabel("pasos de entorno")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("ALE/SpaceInvaders-v5 - comparacion de experimentos", fontsize=13)
    fig.tight_layout()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    destino = FIGURAS / args.salida
    fig.savefig(destino, dpi=140)
    print(f"  Figura -> {destino}")


if __name__ == "__main__":
    main()
