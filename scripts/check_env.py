

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from si_rl.envs import DEFAULT_CONFIG, EnvConfig, describe_env, make_single_env, make_vector_env

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def seccion(titulo: str) -> None:
    print(f"\n{'=' * 78}\n{titulo}\n{'=' * 78}")


def versiones() -> dict:
    seccion("1. VERSIONES Y HARDWARE")
    import ale_py
    import gymnasium
    import torch

    info = {
        "python": __import__("sys").version.split()[0],
        "gymnasium": gymnasium.__version__,
        "ale_py": ale_py.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_disponible": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }
    for k, v in info.items():
        print(f"  {k:22s} {v}")
    if not info["cuda_disponible"]:
        print("\n  ADVERTENCIA: CUDA no disponible. El entrenamiento sera inviable.")
    return info


def metadatos_entorno() -> dict:
    seccion("2. ENTORNO ALE/SpaceInvaders-v5")
    info = describe_env()
    for k, v in info.items():
        print(f"  {k:26s} {v}")

    print("\n  Configuracion canonica del proyecto (EnvConfig):")
    for k, v in dataclasses.asdict(DEFAULT_CONFIG).items():
        print(f"    {k:32s} {v}")

    sticky = info["sticky_action_prob"]
    if abs(sticky - DEFAULT_CONFIG.repeat_action_probability) > 1e-9:
        raise SystemExit(
            f"MISMATCH: el env usa sticky={sticky} pero EnvConfig declara "
            f"{DEFAULT_CONFIG.repeat_action_probability}"
        )
    print(f"\n  OK: sticky actions = {sticky} coincide en ambos caminos.")
    return info


def equivalencia_de_caminos(n_steps: int = 400) -> dict:
    seccion("3. EQUIVALENCIA vector env  <->  gym.make (train/eval match)")

    cfg = dataclasses.replace(
        DEFAULT_CONFIG, repeat_action_probability=0.0, noop_max=0
    )
    print(f"  Config de prueba: sticky=0.0, noop_max=0 (determinista), {n_steps} steps")

    vec = make_vector_env(cfg, num_envs=1, episodic_life=False, reward_clipping=False)
    single = make_single_env(cfg)

    obs_vec, _ = vec.reset(seed=123)
    obs_single, _ = single.reset(seed=123)

    rng = np.random.default_rng(0)
    acciones = rng.integers(0, cfg.n_actions, size=n_steps)

    diffs, iguales = [], 0
    reward_vec = reward_single = 0.0
    comparados = 0

    for i, a in enumerate(acciones):
        ov, rv, tv, uv, _ = vec.step(np.array([a]))
        os_, rs, ts, us, _ = single.step(int(a))

        reward_vec += float(rv[0])
        reward_single += float(rs)

        a_vec = np.asarray(ov[0], dtype=np.int16)
        a_single = np.asarray(os_, dtype=np.int16)
        if a_vec.shape != a_single.shape:
            raise SystemExit(f"shapes distintas: {a_vec.shape} vs {a_single.shape}")

        diff = np.abs(a_vec - a_single)
        diffs.append(float(diff.mean()))
        iguales += int(diff.max() == 0)
        comparados += 1

        if bool(tv[0]) or bool(uv[0]) or ts or us:
            print(f"  (episodio termino en el step {i})")
            break

    vec.close()
    single.close()

    frac_identicas = iguales / comparados
    diff_media = float(np.mean(diffs))
    print(f"  steps comparados            {comparados}")
    print(f"  observaciones identicas     {iguales}/{comparados} ({frac_identicas:.1%})")
    print(f"  diferencia absoluta media   {diff_media:.4f} / 255")
    print(f"  score vector env            {reward_vec}")
    print(f"  score single env            {reward_single}")

    veredicto = "IDENTICOS" if frac_identicas == 1.0 else (
        "EQUIVALENTES" if diff_media < 1.0 else "DIVERGEN"
    )
    print(f"\n  Veredicto: {veredicto}")
    if veredicto == "DIVERGEN":
        print("  Hay que alinear el preprocesamiento antes de entrenar.")

    return {
        "steps_comparados": comparados,
        "fraccion_identicas": frac_identicas,
        "diff_abs_media": diff_media,
        "score_vector": reward_vec,
        "score_single": reward_single,
        "veredicto": veredicto,
    }


def throughput(num_envs_list=(16, 32, 64, 96)) -> dict:
    seccion("4. THROUGHPUT DEL ENTORNO VECTORIZADO")
    resultados = {}
    for n in num_envs_list:
        env = make_vector_env(DEFAULT_CONFIG, num_envs=n,
                              episodic_life=True, reward_clipping=True)
        env.reset(seed=0)
        acciones = np.random.randint(0, DEFAULT_CONFIG.n_actions, size=n)
        for _ in range(50):
            env.step(acciones)

        N = 400
        t0 = time.perf_counter()
        for _ in range(N):
            env.step(acciones)
        dt = time.perf_counter() - t0
        sps = N * n / dt
        env.close()

        resultados[n] = round(sps, 1)
        print(f"  num_envs={n:4d}  {sps:9.0f} steps/s  ({sps * 4:10.0f} frames/s)  "
              f"|  50M steps en {50e6 / sps / 3600:5.2f} h")
    return resultados


def main() -> None:
    reporte = {
        "versiones": versiones(),
        "entorno": metadatos_entorno(),
        "equivalencia": equivalencia_de_caminos(),
        "throughput_steps_por_segundo": throughput(),
        "env_config": dataclasses.asdict(DEFAULT_CONFIG),
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    destino = REPORTS / "etapa0_check_env.json"
    destino.write_text(json.dumps(reporte, indent=2, default=str), encoding="utf-8")
    seccion("RESULTADO")
    print(f"  Reporte guardado en {destino}")


if __name__ == "__main__":
    main()
