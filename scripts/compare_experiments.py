from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from si_rl.evaluate import evaluate_vectorized
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "runs"
REPORTS = RAIZ / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final.pt",
                        help="nombre del archivo dentro de cada carpeta de run")
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--salida", default=None)
    args = parser.parse_args()

    nombres = args.runs or sorted(d.name for d in RUNS.iterdir() if d.is_dir())
    filas, reportes = [], {}

    for nombre in nombres:
        ruta = RUNS / nombre / args.checkpoint
        if not ruta.exists():
            print(f"  [saltado] {nombre}: no existe {args.checkpoint}")
            continue

        politica, env_cfg, meta = load_agent(ruta, epsilon=args.epsilon)
        tcfg = meta["train_config"]
        reporte = evaluate_vectorized(
            env_cfg, politica, label=nombre, n_episodes=args.episodios,
            num_envs=args.num_envs, base_seed=args.seed,
        )
        reportes[nombre] = reporte.summary()
        est = reporte.expected_best_of()

        filas.append({
            "experimento": nombre,
            "algo": tcfg.get("algo", "dqn"),
            "double": tcfg.get("double_dqn", False),
            "dueling": tcfg.get("dueling", False),
            "n_step": tcfg.get("n_step", 1),
            "per": tcfg.get("per", False),
            "steps": meta["step"],
            "media": round(reporte.mean, 1),
            "mediana": round(reporte.median, 1),
            "desv": round(reporte.std, 1),
            "p90": round(reporte.percentile(90), 1),
            "max": reporte.max_score,
            "E_max_de_5": round(est["mean"], 1),
        })
        print(f"  {reporte}", flush=True)

    if not filas:
        raise SystemExit("no se evaluo ningun checkpoint")

    df = pd.DataFrame(filas).sort_values("E_max_de_5", ascending=False)
    REPORTS.mkdir(parents=True, exist_ok=True)
    base = args.salida or f"comparacion_{Path(args.checkpoint).stem}"
    df.to_csv(REPORTS / f"{base}.csv", index=False)
    (REPORTS / f"{base}.json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "episodios": args.episodios,
                    "epsilon": args.epsilon, "reportes": reportes}, indent=2),
        encoding="utf-8")

    print(f"\n{'=' * 100}")
    print(df.to_string(index=False))
    print(f"{'=' * 100}")
    print(f"\n  Tabla -> {REPORTS / (base + '.csv')}")
    print(f"  Ganador por E[max de 5]: {df.iloc[0]['experimento']} "
          f"({df.iloc[0]['E_max_de_5']:.0f})")


if __name__ == "__main__":
    main()
