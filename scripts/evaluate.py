from __future__ import annotations

import argparse
import json
from pathlib import Path

from si_rl.envs import make_single_env
from si_rl.evaluate import COMPETITION_EPISODES, evaluate, record_episode
from si_rl.policies import load_agent

RAIZ = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodios", type=int, default=COMPETITION_EPISODES)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.001,
                        help="exploracion residual; 0.0 es greedy puro")
    parser.add_argument("--sin-video", action="store_true")
    parser.add_argument("--salida", type=str, default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    politica, env_cfg, meta = load_agent(ckpt, epsilon=args.epsilon)

    print(f"\n{'=' * 78}")
    print(f"  checkpoint     {ckpt}")
    print(f"  entrenado a    {meta['step']:,} steps")
    print(f"  entorno        {env_cfg.env_id}  sticky={env_cfg.repeat_action_probability}")
    print(f"  observacion    {env_cfg.obs_shape}")
    print(f"  epsilon eval   {args.epsilon}")
    print(f"{'=' * 78}\n")

    reporte = evaluate(
        lambda: make_single_env(env_cfg), politica,
        label=ckpt.stem, n_episodes=args.episodios, base_seed=args.seed,
        metadata={"checkpoint": str(ckpt), "step": meta["step"],
                  "epsilon": args.epsilon, "env_config": env_cfg.to_dict()},
    )

    print(f"\n{'=' * 78}")
    print(f"  media          {reporte.mean:10.1f}")
    print(f"  mediana        {reporte.median:10.1f}")
    print(f"  desv. est.     {reporte.std:10.1f}")
    print(f"  MAXIMO         {reporte.max_score:10.1f}   <- metrica de la competencia")
    if args.episodios > COMPETITION_EPISODES:
        est = reporte.expected_best_of()
        print(f"  E[max de 5]    {est['mean']:10.1f}   (p10 {est['p10']:.0f} / "
              f"p90 {est['p90']:.0f})")
    print(f"{'=' * 78}")

    salida = Path(args.salida) if args.salida else RAIZ / "reports" / f"eval_{ckpt.stem}.json"
    reporte.to_json(salida)
    print(f"\n  Reporte -> {salida}")

    if not args.sin_video:
        mejor = max(reporte.episodes, key=lambda e: e.score)
        carpeta = RAIZ / "reports" / "videos" / ckpt.stem
        res = record_episode(
            lambda: make_single_env(env_cfg, video_folder=str(carpeta),
                                    name_prefix=ckpt.stem),
            politica, seed=mejor.seed,
        )
        print(f"  Video (seed {mejor.seed}, score {res.score:.0f}) -> {carpeta}")
        (carpeta / "metadata.json").write_text(
            json.dumps({"checkpoint": str(ckpt), "step": meta["step"],
                        "seed": mejor.seed, "score": res.score,
                        "steps": res.steps, "epsilon": args.epsilon,
                        "env_config": env_cfg.to_dict()}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
