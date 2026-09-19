from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "runs"


def cargar(run_dir: Path) -> EventAccumulator:
    acc = EventAccumulator(str(run_dir / "tensorboard"),
                           size_guidance={"scalars": 100_000})
    acc.Reload()
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", default=None)
    parser.add_argument("--ultimos", type=int, default=12)
    args = parser.parse_args()

    if args.run:
        run_dir = RUNS / args.run
    else:
        candidatos = [d for d in RUNS.iterdir() if (d / "tensorboard").exists()]
        if not candidatos:
            raise SystemExit("no hay corridas en runs/")
        run_dir = max(candidatos, key=lambda d: d.stat().st_mtime)

    acc = cargar(run_dir)
    etiquetas = acc.Tags()["scalars"]
    print(f"\n  corrida: {run_dir.name}")
    print(f"  escalares: {', '.join(etiquetas)}\n")

    def serie(tag):
        return acc.Scalars(tag) if tag in etiquetas else []

    progreso = serie("entrenamiento/sps")
    if progreso:
        ultimo = progreso[-1]
        print(f"  step actual      {ultimo.step:,}")
        print(f"  sps              {ultimo.value:.0f}")

    for tag, nombre in [("episodio/score_medio_200", "score entrenamiento (200 ep)"),
                        ("entrenamiento/q_medio", "Q medio"),
                        ("entrenamiento/perdida", "perdida"),
                        ("entrenamiento/epsilon", "epsilon")]:
        s = serie(tag)
        if s:
            print(f"  {nombre:28s} {s[-1].value:10.3f}")

    evals = serie("eval/score_medio")
    if evals:
        print(f"\n  {'step':>12}  {'eval media':>11}  {'eval max':>10}  {'E[max de 5]':>12}")
        maxs = {e.step: e.value for e in serie("eval/score_max")}
        b5 = {e.step: e.value for e in serie("eval/best_of_5")}
        for e in evals[-args.ultimos:]:
            print(f"  {e.step:>12,}  {e.value:>11.1f}  {maxs.get(e.step, 0):>10.1f}  "
                  f"{b5.get(e.step, 0):>12.1f}")

    episodios = serie("episodio/score")
    if episodios:
        valores = [e.value for e in episodios]
        print(f"\n  episodios registrados {len(valores):,}  "
              f"max historico {max(valores):.0f}")


if __name__ == "__main__":
    main()
