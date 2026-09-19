from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "checkpoints"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--distorsion", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--nota", type=str, default="")
    args = parser.parse_args()

    origen = Path(args.checkpoint)
    if not origen.exists():
        raise SystemExit(f"no existe: {origen}")

    import ale_py, gymnasium, numpy, torch
    from si_rl.policies import load_agent

    politica, env_cfg, meta = load_agent(origen, epsilon=args.epsilon)

    DESTINO.mkdir(parents=True, exist_ok=True)
    destino_pt = DESTINO / "champion.pt"
    shutil.copy2(origen, destino_pt)

    ficha = {
        "congelado": datetime.now().isoformat(timespec="seconds"),
        "origen": str(origen),
        "sha256": sha256(destino_pt),
        "pasos_de_entrenamiento": meta["step"],
        "train_config": meta["train_config"],
        "env_config": env_cfg.to_dict(),
        "politica_de_evaluacion": {
            "epsilon": args.epsilon,
            "distorsion": args.distorsion,
            "eta": args.eta if args.distorsion else None,
        },
        "eval_al_guardar": meta.get("extra", {}).get("eval"),
        "versiones": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gymnasium": gymnasium.__version__,
            "ale_py": ale_py.__version__,
            "numpy": numpy.__version__,
        },
        "nota": args.nota,
    }
    (DESTINO / "champion.json").write_text(json.dumps(ficha, indent=2), encoding="utf-8")

    print(f"\n  Campeon congelado")
    print(f"    pesos     {destino_pt}")
    print(f"    ficha     {DESTINO / 'champion.json'}")
    print(f"    origen    {origen}  ({meta['step']:,} steps)")
    print(f"    sha256    {ficha['sha256'][:16]}...")
    print(f"\n  Para evaluar:\n    python scripts/evaluate.py checkpoints/champion.pt")


if __name__ == "__main__":
    main()
