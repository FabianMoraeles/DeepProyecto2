from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from si_rl.envs import DEFAULT_CONFIG
from si_rl.train_dqn import TrainConfig, train

RAIZ = Path(__file__).resolve().parents[1]

ABLACIONES = {
    "E01_dqn": {},
    "E02_double": {"double_dqn": True},
    "E03_dueling": {"double_dqn": True, "dueling": True},
    "E04_nstep": {"double_dqn": True, "dueling": True, "n_step": 3},
    "E05_per": {"double_dqn": True, "dueling": True, "n_step": 3, "per": True},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo", nargs="+", default=["E02_double", "E03_dueling",
                                                      "E04_nstep", "E05_per"])
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    args = parser.parse_args()

    resumen_path = RAIZ / "reports" / "ablaciones_fase2.json"
    resumen = json.loads(resumen_path.read_text(encoding="utf-8")) if resumen_path.exists() else {}

    for exp_id in args.solo:
        if exp_id not in ABLACIONES:
            raise SystemExit(f"experimento desconocido: {exp_id}. "
                             f"Opciones: {list(ABLACIONES)}")

        cfg = TrainConfig(
            exp_id=exp_id, seed=args.seed, total_steps=args.total_steps,
            buffer_size=args.buffer_size, lr=args.lr, **ABLACIONES[exp_id],
        )
        print(f"\n\n{'#' * 78}\n#  {exp_id}\n{'#' * 78}", flush=True)
        try:
            resultado = train(cfg, DEFAULT_CONFIG)
        except Exception:
            print(f"  {exp_id} fallo:\n{traceback.format_exc()}", flush=True)
            resumen[exp_id] = {"error": traceback.format_exc()[-2000:]}
        else:
            resumen[exp_id] = {
                "config": asdict(cfg),
                "steps": resultado["steps"],
                "duracion_min": round(resultado["duracion_s"] / 60, 1),
                "mejor_eval_media": resultado["mejor_eval_media"],
                "historial": resultado["historial"],
            }

        resumen_path.parent.mkdir(parents=True, exist_ok=True)
        resumen_path.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
        print(f"\n  Resumen acumulado -> {resumen_path}", flush=True)


if __name__ == "__main__":
    main()
