from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from si_rl.evaluate import evaluate_vectorized
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "runs"
REPORTS = RAIZ / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str)
    parser.add_argument("--episodios", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=31_000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--desde", type=int, default=0,
                        help="ignora checkpoints anteriores a este paso")
    parser.add_argument("--incluir-mejor", action="store_true", default=True)
    args = parser.parse_args()

    run_dir = RUNS / args.run
    if not run_dir.exists():
        raise SystemExit(f"no existe {run_dir}")

    candidatos = []
    for ruta in sorted(run_dir.glob("step_*.pt")):
        m = re.search(r"step_(\d+)\.pt", ruta.name)
        if m and int(m.group(1)) >= args.desde:
            candidatos.append(ruta)
    for extra in ("mejor.pt", "final.pt"):
        if (run_dir / extra).exists():
            candidatos.append(run_dir / extra)

    if not candidatos:
        raise SystemExit("no hay checkpoints que evaluar")

    print(f"  {len(candidatos)} checkpoints, {args.episodios} episodios cada uno, "
          f"epsilon={args.epsilon}\n", flush=True)

    filas = []
    for ruta in candidatos:
        politica, env_cfg, meta = load_agent(ruta, epsilon=args.epsilon)
        reporte = evaluate_vectorized(
            env_cfg, politica, label=ruta.name, n_episodes=args.episodios,
            num_envs=args.num_envs, base_seed=args.seed,
        )
        est = reporte.expected_best_of()
        filas.append({
            "checkpoint": ruta.name,
            "steps": meta["step"],
            "media": round(reporte.mean, 1),
            "mediana": round(reporte.median, 1),
            "desv": round(reporte.std, 1),
            "p90": round(reporte.percentile(90), 1),
            "max": reporte.max_score,
            "E_max_de_5": round(est["mean"], 1),
        })
        print(f"  {reporte}", flush=True)

    df = pd.DataFrame(filas).sort_values("E_max_de_5", ascending=False)
    print(f"\n{'=' * 95}")
    print(df.to_string(index=False))
    print(f"{'=' * 95}")

    mejor = df.iloc[0]
    se = mejor["desv"] / (args.episodios ** 0.5)
    print(f"\n  Mejor: {mejor['checkpoint']} ({mejor['steps']:,} steps)")
    print(f"    E[max de 5] = {mejor['E_max_de_5']:.0f}   media = {mejor['media']:.0f} +- {se:.0f}")
    segundo = df.iloc[1]
    print(f"    segundo: {segundo['checkpoint']} con {segundo['E_max_de_5']:.0f}"
          f"  (diferencia {mejor['E_max_de_5'] - segundo['E_max_de_5']:.0f})")

    REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORTS / f"seleccion_checkpoint_{args.run}.csv", index=False)
    (REPORTS / f"seleccion_checkpoint_{args.run}.json").write_text(
        json.dumps({"run": args.run, "episodios": args.episodios,
                    "epsilon": args.epsilon, "resultados": filas}, indent=2),
        encoding="utf-8")
    print(f"\n  Tabla -> {REPORTS / f'seleccion_checkpoint_{args.run}.csv'}")


if __name__ == "__main__":
    main()
