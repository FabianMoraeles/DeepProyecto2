from __future__ import annotations

import argparse
import json
from pathlib import Path

from si_rl.evaluate import evaluate_vectorized
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodios", type=int, default=50)
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.0, 0.001, 0.01, 0.05])
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=25)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    filas = []

    for eps in args.epsilons:
        politica, env_cfg, meta = load_agent(ckpt, epsilon=eps)
        reporte = evaluate_vectorized(
            env_cfg, politica, label=f"eps={eps}", n_episodes=args.episodios,
            num_envs=args.num_envs, base_seed=args.seed,
        )
        est = reporte.expected_best_of()
        filas.append({
            "epsilon": eps,
            "media": round(reporte.mean, 1),
            "mediana": round(reporte.median, 1),
            "desv": round(reporte.std, 1),
            "p90": round(reporte.percentile(90), 1),
            "max": reporte.max_score,
            "E_max_de_5": round(est["mean"], 1),
            "p10_max_de_5": round(est["p10"], 1),
        })
        print(f"  eps={eps:<6} {reporte}", flush=True)

    print(f"\n{'epsilon':>8} {'media':>9} {'desv':>8} {'p90':>9} {'max':>9} {'E[max5]':>10}")
    for f in filas:
        print(f"{f['epsilon']:>8} {f['media']:>9.1f} {f['desv']:>8.1f} "
              f"{f['p90']:>9.1f} {f['max']:>9.1f} {f['E_max_de_5']:>10.1f}")

    mejor = max(filas, key=lambda f: f["E_max_de_5"])
    print(f"\n  Mejor para la competencia: epsilon={mejor['epsilon']} "
          f"con E[max de 5]={mejor['E_max_de_5']:.1f}")

    salida = RAIZ / "reports" / f"sweep_epsilon_{ckpt.stem}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"checkpoint": str(ckpt), "episodios": args.episodios,
                                  "resultados": filas}, indent=2), encoding="utf-8")
    print(f"  Reporte -> {salida}")


if __name__ == "__main__":
    main()
