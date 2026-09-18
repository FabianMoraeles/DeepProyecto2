"""Entrenamiento DQN con entornos vectorizados.

Decisiones que difieren de un DQN de libro, y por que:

  - episodic_life desactivado. ALE no expone si una terminacion fue perdida de
    vida o fin de partida, asi que se llevan las vidas a mano con info['lives'].
    A cambio el score real por episodio sale gratis y correcto.
  - reward_clipping desactivado en el entorno y aplicado aqui con np.sign. El
    entorno entrega el score real, que es lo que se registra; la red entrena
    con la senal acotada.
  - 64 entornos en paralelo con batch grande y una actualizacion por paso
    vectorizado. La razon de replay resultante (8 muestras por transicion) es
    la misma del DQN de Nature, pero con el entorno a 15k steps/s.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from si_rl.agents import make_agent
from si_rl.envs import DEFAULT_CONFIG, EnvConfig, make_vector_env
from si_rl.evaluate import evaluate_vectorized
from si_rl.policies import GreedyPolicy, save_checkpoint
from si_rl.replay import PrefetchSampler, PrioritizedReplayBuffer, ReplayBuffer

RAIZ = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    exp_id: str = "E01_dqn"
    seed: int = 0
    total_steps: int = 10_000_000          # transiciones de entorno
    num_envs: int = 64

    # algoritmo y componentes (ablaciones de las fases 2 y 3)
    algo: str = "dqn"                      # "dqn" o "iqn"
    double_dqn: bool = False
    dueling: bool = False
    per: bool = False

    # IQN
    iqn_n: int = 8                         # cuantiles de la red en linea
    iqn_n_prime: int = 8                   # cuantiles del objetivo
    iqn_k: int = 32                        # cuantiles para estimar Q al actuar
    iqn_n_cos: int = 64
    iqn_kappa: float = 1.0

    # replay
    buffer_size: int = 500_000
    learning_starts: int = 80_000
    batch_size: int = 512
    n_step: int = 1
    per_alpha: float = 0.5
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # Tratamiento de la recompensa antes de guardarla en el replay:
    #   "clip" - np.sign(r), el estandar de DQN
    #   "h"    - recompensa cruda, con reescalado invertible en el objetivo
    #   "raw"  - recompensa cruda sin reescalar (inestable, solo para control)
    reward_transform: str = "clip"

    # optimizacion
    lr: float = 1e-4
    adam_eps: float = 1.5e-4
    # 0.99 da un horizonte efectivo de ~100 pasos, pero los episodios duran
    # ~2000: el agente apenas percibe el costo de morir. 0.997 lo lleva a ~333.
    gamma: float = 0.99
    target_update_interval: int = 40_000   # en transiciones de entorno
    max_grad_norm: float = 10.0
    updates_per_step: int = 1              # por paso vectorizado

    # exploracion
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 1_000_000
    eval_epsilon: float = 0.001

    # logging / evaluacion
    log_interval: int = 20_000
    eval_interval: int = 500_000
    # Con 10 episodios la desviacion del score es tan grande que elegir el mejor
    # checkpoint por la media selecciona suerte. La evaluacion vectorizada hace
    # que 50 cuesten menos que 10 por la ruta de un solo entorno.
    eval_episodes: int = 50
    eval_envs: int = 32
    checkpoint_interval: int = 1_000_000

    # Reanudacion: ruta a un checkpoint del que continuar. El replay buffer no
    # se guarda, asi que la corrida vuelve a llenarlo desde cero antes de
    # entrenar; el contador de pasos si continua donde estaba.
    resume: str = ""

    device: str = "cuda"
    # La CNN de Nature es demasiado pequena para que los tensor cores compensen
    # el costo de autocast: medido, bf16 tarda 4.11 ms por update contra 3.39 ms
    # en fp32 con TF32. Se deja el interruptor por si crece la arquitectura.
    amp: bool = False
    prefetch: bool = True                  # muestreo del replay en un hilo aparte

    def replay_ratio(self) -> float:
        return self.updates_per_step * self.batch_size / self.num_envs


@dataclass
class EpisodeTracker:
    """Score real y duracion por entorno, con estadisticas moviles."""

    num_envs: int
    score: np.ndarray = field(init=False)
    length: np.ndarray = field(init=False)
    lives_lost: np.ndarray = field(init=False)
    recientes: list[float] = field(default_factory=list)
    total_episodios: int = 0

    def __post_init__(self):
        self.score = np.zeros(self.num_envs, dtype=np.float64)
        self.length = np.zeros(self.num_envs, dtype=np.int64)
        self.lives_lost = np.zeros(self.num_envs, dtype=np.int64)

    def update(self, rewards: np.ndarray, activo: np.ndarray, perdio_vida: np.ndarray,
               fin: np.ndarray) -> list[float]:
        self.score += np.where(activo, rewards, 0.0)
        self.length += activo.astype(np.int64)
        self.lives_lost += (perdio_vida & activo).astype(np.int64)

        terminados = []
        for e in np.flatnonzero(fin):
            terminados.append(float(self.score[e]))
            self.recientes.append(float(self.score[e]))
            self.total_episodios += 1
            self.score[e] = 0.0
            self.length[e] = 0
            self.lives_lost[e] = 0
        if len(self.recientes) > 200:
            self.recientes = self.recientes[-200:]
        return terminados

    @property
    def media_reciente(self) -> float:
        return float(np.mean(self.recientes)) if self.recientes else 0.0

    @property
    def max_reciente(self) -> float:
        return float(np.max(self.recientes)) if self.recientes else 0.0


def epsilon_en(step: int, cfg: TrainConfig) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    frac = step / cfg.eps_decay_steps
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train(cfg: TrainConfig, env_cfg: EnvConfig = DEFAULT_CONFIG) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = RAIZ / "runs" / f"{cfg.exp_id}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "env": env_cfg.to_dict()}, indent=2),
        encoding="utf-8",
    )
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    env = make_vector_env(env_cfg, num_envs=cfg.num_envs,
                          episodic_life=False, reward_clipping=False)
    obs, info = env.reset(seed=cfg.seed)

    comun = dict(
        capacity=cfg.buffer_size, num_envs=cfg.num_envs,
        stack=env_cfg.stack_num, height=env_cfg.img_height, width=env_cfg.img_width,
        gamma=cfg.gamma, n_step=cfg.n_step, seed=cfg.seed,
    )
    if cfg.per:
        buffer = PrioritizedReplayBuffer(
            **comun, alpha=cfg.per_alpha, beta_start=cfg.per_beta_start,
            beta_end=cfg.per_beta_end, beta_steps=cfg.total_steps,
        )
    else:
        buffer = ReplayBuffer(**comun)

    n_actions = env_cfg.n_actions
    agente = make_agent(cfg, env_cfg, device)
    net = agente.net

    paso_inicial = 0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        agente.net.load_state_dict(ckpt["model_state_dict"])
        agente.sync_target()
        if ckpt.get("optimizer_state_dict"):
            agente.opt.load_state_dict(ckpt["optimizer_state_dict"])
            estado_opt = "con estado de Adam"
        else:
            estado_opt = "con Adam reiniciado"
        paso_inicial = int(ckpt["step"])
        print(f"  reanudando desde {cfg.resume} en el paso {paso_inicial:,} {estado_opt}")

    def guardar(path, **kw):
        save_checkpoint(path, net=net, env_config=env_cfg, train_config=asdict(cfg),
                        arch=agente.arch, net_kwargs=agente.net_kwargs,
                        optimizer=agente.opt, **kw)

    rng = np.random.default_rng(cfg.seed)
    tracker = EpisodeTracker(cfg.num_envs)
    prev_fin = np.zeros(cfg.num_envs, dtype=bool)
    vidas_prev = np.asarray(info["lives"]).copy()

    componentes = [cfg.algo.upper()] + [
        n for n, on in [("double", cfg.double_dqn), ("dueling", cfg.dueling),
                        (f"n-step={cfg.n_step}", cfg.n_step > 1),
                        ("PER", cfg.per),
                        (f"reward={cfg.reward_transform}", cfg.reward_transform != "clip"),
                        (f"gamma={cfg.gamma}", cfg.gamma != 0.99)] if on
    ]
    print(f"\n{'=' * 78}")
    print(f"  {cfg.exp_id}  seed={cfg.seed}  device={device}")
    print(f"  componentes: {' + '.join(componentes)}")
    print(f"  buffer {buffer.capacity:,} transiciones = {buffer.nbytes / 2**30:.2f} GB")
    print(f"  razon de replay {cfg.replay_ratio():.1f} muestras por transicion")
    print(f"{'=' * 78}\n", flush=True)

    sampler: PrefetchSampler | None = None
    global_step = paso_inicial
    # Pasos de esta corrida: el calentamiento del replay depende de cuanto se
    # lleva recolectado ahora, no del contador global heredado del checkpoint.
    pasos_propios = 0
    updates = 0
    ultimo_log = global_step
    ultimo_eval = global_step
    ultimo_ckpt = global_step
    ultimo_target = global_step
    t_inicio = time.perf_counter()
    t_ventana = t_inicio
    step_ventana = global_step
    perdidas: list[float] = []
    q_medios: list[float] = []
    mejor_eval = -np.inf
    historial: list[dict] = []

    while global_step < cfg.total_steps:
        eps = epsilon_en(global_step, cfg)

        if pasos_propios < cfg.learning_starts and not cfg.resume:
            acciones = rng.integers(0, n_actions, size=cfg.num_envs)
        else:
            acciones = agente.act(obs, eps, rng)

        next_obs, recompensas, term, trunc, info = env.step(acciones)
        fin = term | trunc
        dummy = prev_fin
        activo = ~dummy

        vidas = np.asarray(info["lives"])
        perdio_vida = (vidas < vidas_prev) & activo
        vidas_prev = vidas.copy()

        # El frame del paso dummy es el primero del episodio nuevo.
        frame = np.where(dummy[:, None, None], next_obs[:, -1], obs[:, -1])
        r_train = (np.sign(recompensas) if cfg.reward_transform == "clip"
                   else recompensas)
        buffer.add(frame, acciones, r_train.astype(np.float32), fin, dummy=dummy)

        terminados = tracker.update(recompensas, activo, perdio_vida, fin)
        for score in terminados:
            writer.add_scalar("episodio/score", score, global_step)

        prev_fin = fin
        obs = next_obs
        global_step += cfg.num_envs
        pasos_propios += cfg.num_envs

        # ------------------------------------------------------------------ #
        if pasos_propios >= cfg.learning_starts:
            if cfg.per:
                buffer.set_beta_por_paso(global_step)
            if cfg.prefetch and sampler is None:
                sampler = PrefetchSampler(buffer, cfg.batch_size)
            for _ in range(cfg.updates_per_step):
                lote = sampler.get() if sampler else buffer.sample(cfg.batch_size)
                res = agente.update(lote)
                updates += 1
                perdidas.append(res.perdida)
                q_medios.append(res.q_medio)
                if cfg.per:
                    buffer.update_priorities(lote.indices, res.prioridades)

        if global_step - ultimo_target >= cfg.target_update_interval:
            agente.sync_target()
            ultimo_target = global_step

        # ------------------------------------------------------------------ #
        if global_step - ultimo_log >= cfg.log_interval:
            ahora = time.perf_counter()
            sps = (global_step - step_ventana) / (ahora - t_ventana)
            t_ventana, step_ventana = ahora, global_step
            ultimo_log = global_step

            perdida_media = float(np.mean(perdidas)) if perdidas else 0.0
            q_medio = float(np.mean(q_medios)) if q_medios else 0.0
            perdidas.clear()
            q_medios.clear()

            writer.add_scalar("entrenamiento/perdida", perdida_media, global_step)
            writer.add_scalar("entrenamiento/q_medio", q_medio, global_step)
            writer.add_scalar("entrenamiento/epsilon", eps, global_step)
            writer.add_scalar("entrenamiento/sps", sps, global_step)
            writer.add_scalar("episodio/score_medio_200", tracker.media_reciente, global_step)

            transcurrido = ahora - t_inicio
            restante = (cfg.total_steps - global_step) / max(sps, 1e-6)
            print(
                f"  step {global_step:>10,}  eps {eps:.3f}  "
                f"score200 {tracker.media_reciente:7.1f}  max {tracker.max_reciente:7.1f}  "
                f"q {q_medio:7.3f}  loss {perdida_media:.4f}  "
                f"{sps:6.0f} sps  t {transcurrido / 60:5.1f}m  eta {restante / 60:5.1f}m",
                flush=True,
            )

        # ------------------------------------------------------------------ #
        if global_step - ultimo_eval >= cfg.eval_interval and pasos_propios >= cfg.learning_starts:
            ultimo_eval = global_step
            politica = GreedyPolicy(net, device, epsilon=cfg.eval_epsilon, seed=cfg.seed)
            reporte = evaluate_vectorized(
                env_cfg, politica, label=f"{cfg.exp_id}@{global_step}",
                n_episodes=cfg.eval_episodes, num_envs=cfg.eval_envs,
                base_seed=10_000,
            )
            net.train()
            writer.add_scalar("eval/score_medio", reporte.mean, global_step)
            writer.add_scalar("eval/score_max", reporte.max_score, global_step)
            writer.add_scalar("eval/best_of_5", reporte.expected_best_of()["mean"], global_step)
            historial.append({"step": global_step, **reporte.summary()})
            print(f"    EVAL  {reporte}", flush=True)

            if reporte.mean > mejor_eval:
                mejor_eval = reporte.mean
                guardar(run_dir / "mejor.pt", step=global_step,
                        extra={"eval": reporte.summary()})

        if global_step - ultimo_ckpt >= cfg.checkpoint_interval:
            ultimo_ckpt = global_step
            guardar(run_dir / f"step_{global_step}.pt", step=global_step)

    if sampler is not None:
        sampler.close()
    env.close()
    guardar(run_dir / "final.pt", step=global_step)
    (run_dir / "historial_eval.json").write_text(
        json.dumps(historial, indent=2), encoding="utf-8")
    writer.close()

    duracion = time.perf_counter() - t_inicio
    print(f"\n  Listo en {duracion / 60:.1f} min  ({global_step:,} steps, {updates:,} updates)")
    return {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "steps": global_step,
        "updates": updates,
        "duracion_s": duracion,
        "mejor_eval_media": mejor_eval,
        "historial": historial,
        "run_dir": str(run_dir),
    }
