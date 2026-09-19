from __future__ import annotations

import argparse
import dataclasses

from si_rl.envs import DEFAULT_CONFIG
from si_rl.train_dqn import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser()
    campos = {f.name: f for f in dataclasses.fields(TrainConfig)}
    for nombre, campo in campos.items():
        flag = "--" + nombre.replace("_", "-")
        if campo.type is bool or isinstance(campo.default, bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "si", "yes"),
                                default=None)
        else:
            parser.add_argument(flag, type=type(campo.default), default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="corrida corta para validar el pipeline")

    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k != "smoke" and v is not None}

    if args.smoke:
        overrides = {
            "exp_id": "E00_smoke",
            "total_steps": 400_000,
            "buffer_size": 100_000,
            "learning_starts": 20_000,
            "log_interval": 20_000,
            "eval_interval": 150_000,
            "eval_episodes": 5,
            "checkpoint_interval": 10_000_000,
            "eps_decay_steps": 150_000,
            **overrides,
        }

    cfg = TrainConfig(**overrides)
    resultado = train(cfg, DEFAULT_CONFIG)
    print(f"\n  Resultados en {resultado['run_dir']}")


if __name__ == "__main__":
    main()
