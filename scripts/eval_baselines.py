from __future__ import annotations

import argparse
from pathlib import Path

from si_rl.baselines import RandomPolicy, RuleBasedPolicy
from si_rl.envs import DEFAULT_CONFIG, describe_env, make_raw_env, make_single_env
from si_rl.evaluate import evaluate, record_episode

RAIZ = Path(__file__).resolve().parents[1]
REPORTS = RAIZ / "reports"
VIDEOS = RAIZ / "reports" / "videos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--sin-video", action="store_true")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG
    meta = {"env_config": cfg.to_dict(), "action_meanings": describe_env()["action_meanings"]}
    reportes = []

    print(f"\n{'=' * 78}\nBASELINE 1/2: politica aleatoria\n{'=' * 78}")
    aleatorio = evaluate(
        lambda: make_single_env(cfg),
        RandomPolicy(cfg.n_actions, seed=0),
        label="aleatorio",
        n_episodes=args.episodios,
        base_seed=args.seed,
        metadata=meta,
    )
    reportes.append(aleatorio)

    print(f"\n{'=' * 78}\nBASELINE 2/2: regla simple (pixeles crudos, Lab 5)\n{'=' * 78}")
    meanings = describe_env()["action_meanings"]
    regla = evaluate(
        lambda: make_raw_env(cfg),
        RuleBasedPolicy(meanings),
        label="regla_simple",
        n_episodes=args.episodios,
        base_seed=args.seed,
        metadata=meta,
    )
    reportes.append(regla)

    print(f"\n{'=' * 78}\nRESUMEN\n{'=' * 78}")
    for r in reportes:
        print(r)
        r.to_json(REPORTS / f"etapa0_baseline_{r.label}.json")

    if not args.sin_video:
        print(f"\n{'=' * 78}\nVIDEOS\n{'=' * 78}")
        mejor_aleatorio = max(aleatorio.episodes, key=lambda e: e.score)
        res = record_episode(
            lambda: make_single_env(cfg, video_folder=str(VIDEOS / "aleatorio"),
                                    name_prefix="aleatorio"),
            RandomPolicy(cfg.n_actions, seed=0),
            seed=mejor_aleatorio.seed,
        )
        print(f"  aleatorio     score={res.score:8.1f}  -> {VIDEOS / 'aleatorio'}")

        mejor_regla = max(regla.episodes, key=lambda e: e.score)
        res = record_episode(
            lambda: make_raw_env(cfg, video_folder=str(VIDEOS / "regla_simple"),
                                 name_prefix="regla_simple"),
            RuleBasedPolicy(meanings),
            seed=mejor_regla.seed,
        )
        print(f"  regla_simple  score={res.score:8.1f}  -> {VIDEOS / 'regla_simple'}")

    print(f"\n  Reportes en {REPORTS}")


if __name__ == "__main__":
    main()
