from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from si_rl.evaluate import evaluate_vectorized
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]

CONFIGURACIONES = [
    ("neutral", 0.0),
    ("cvar_sup", 0.75),
    ("cvar_sup", 0.50),
    ("cvar_sup", 0.25),
    ("wang", 0.75),
    ("pow", 1.0),
    ("cvar", 0.50),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    politica, env_cfg, meta = load_agent(ckpt, epsilon=args.epsilon)
    if not hasattr(politica.net, "distorsionar"):
        raise SystemExit("este checkpoint no es IQN: no admite distorsiones de riesgo")

    print(f"  checkpoint {ckpt}  ({meta['step']:,} steps)\n")
    filas = []
    for modo, eta in CONFIGURACIONES:
        politica.net.distortion = modo
        politica.net.distortion_eta = eta
        reporte = evaluate_vectorized(
            env_cfg, politica, label=f"{modo}({eta})", n_episodes=args.episodios,
            num_envs=args.num_envs, base_seed=args.seed,
        )
        est = reporte.expected_best_of()
        filas.append({
            "distorsion": modo, "eta": eta,
            "media": round(reporte.mean, 1),
            "desv": round(reporte.std, 1),
            "p90": round(reporte.percentile(90), 1),
            "max": reporte.max_score,
            "E_max_de_5": round(est["mean"], 1),
        })
        print(f"  {reporte}", flush=True)

    df = pd.DataFrame(filas).sort_values("E_max_de_5", ascending=False)
    print(f"\n{'=' * 90}")
    print(df.to_string(index=False))
    print(f"{'=' * 90}")
    mejor = df.iloc[0]
    print(f"\n  Mejor para max-de-5: {mejor['distorsion']}(eta={mejor['eta']}) "
          f"-> E[max de 5] = {mejor['E_max_de_5']:.0f}  "
          f"(neutral: {df[df.distorsion == 'neutral'].iloc[0]['E_max_de_5']:.0f})")

    salida = RAIZ / "reports" / f"sweep_risk_{ckpt.stem}"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{salida}.csv", index=False)
    Path(f"{salida}.json").write_text(
        json.dumps({"checkpoint": str(ckpt), "step": meta["step"],
                    "episodios": args.episodios, "resultados": filas}, indent=2),
        encoding="utf-8")
    print(f"  Tabla -> {salida}.csv")


if __name__ == "__main__":
    main()
