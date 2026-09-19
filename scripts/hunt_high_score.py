from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from si_rl.envs import make_single_env
from si_rl.evaluate import EvalReport, record_episode, run_episode
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "data" / "evaluation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--intentos", type=int, default=300)
    parser.add_argument("--seed-inicial", type=int, default=100_000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--distorsion", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--sin-video", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    politica, env_cfg, meta = load_agent(ckpt, epsilon=args.epsilon)
    if args.distorsion:
        if not hasattr(politica.net, "distorsionar"):
            raise SystemExit("este checkpoint no es IQN: no admite distorsiones")
        politica.net.distortion = args.distorsion
        politica.net.distortion_eta = args.eta

    etiqueta = ckpt.stem + (f"_{args.distorsion}{args.eta}" if args.distorsion else "")
    print(f"\n  checkpoint  {ckpt}  ({meta['step']:,} steps)")
    print(f"  politica    epsilon={args.epsilon} distorsion={args.distorsion or 'neutral'}")
    print(f"  intentos    {args.intentos}\n")

    env = make_single_env(env_cfg)
    registros, mejor, mejor_seed = [], -1.0, None
    t0 = time.perf_counter()
    try:
        for i in range(args.intentos):
            seed = args.seed_inicial + i
            res = run_episode(env, politica, seed=seed, episode=i)
            registros.append({"intento": i, "seed": seed, "score": res.score,
                              "steps": res.steps, "duracion_s": round(res.duration_s, 2)})
            if res.score > mejor:
                mejor, mejor_seed = res.score, seed
                print(f"  [{i:4d}] NUEVO RECORD  score={res.score:8.1f}  seed={seed}",
                      flush=True)
            elif (i + 1) % 25 == 0:
                print(f"  [{i:4d}] ... mejor={mejor:.0f}  "
                      f"({(i + 1) / (time.perf_counter() - t0):.1f} ep/s)", flush=True)
    finally:
        env.close()

    df = pd.DataFrame(registros)
    reporte = EvalReport(label=etiqueta, n_episodes=len(df),
                         scores=df["score"].tolist(),
                         metadata={"checkpoint": str(ckpt), "step": meta["step"],
                                   "epsilon": args.epsilon,
                                   "distorsion": args.distorsion, "eta": args.eta,
                                   "env_config": env_cfg.to_dict()})
    est = reporte.expected_best_of()

    print(f"\n{'=' * 78}")
    print(f"  episodios      {len(df)}")
    print(f"  media          {reporte.mean:10.1f}")
    print(f"  mediana        {reporte.median:10.1f}")
    print(f"  desv. est.     {reporte.std:10.1f}")
    print(f"  p90            {reporte.percentile(90):10.1f}")
    print(f"  E[max de 5]    {est['mean']:10.1f}")
    print(f"  RECORD         {mejor:10.1f}   (seed {mejor_seed})")
    print(f"{'=' * 78}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    df.sort_values("score", ascending=False).to_csv(
        SALIDA / f"leaderboard_{etiqueta}.csv", index=False)
    reporte.to_json(SALIDA / f"hunt_{etiqueta}.json")

    if not args.sin_video and mejor_seed is not None:
        carpeta = RAIZ / "reports" / "videos" / f"record_{etiqueta}"
        res = record_episode(
            lambda: make_single_env(env_cfg, video_folder=str(carpeta),
                                    name_prefix=f"record_{etiqueta}"),
            politica, seed=mejor_seed,
        )
        coincide = abs(res.score - mejor) < 1e-6
        print(f"\n  Video grabado: score={res.score:.0f} (esperado {mejor:.0f})"
              f"{'  OK reproducible' if coincide else '  AVISO: no coincide'}")
        print(f"  -> {carpeta}")
        (carpeta / "metadata.json").write_text(json.dumps({
            "checkpoint": str(ckpt), "step": meta["step"], "seed": mejor_seed,
            "score": res.score, "score_esperado": mejor, "reproducible": coincide,
            "steps": res.steps, "epsilon": args.epsilon,
            "distorsion": args.distorsion, "eta": args.eta,
            "env_config": env_cfg.to_dict(),
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
