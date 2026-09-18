"""Fuente unica de verdad para la configuracion del entorno.

Entrenamiento, evaluacion y grabacion de video deben construir sus entornos
desde aqui. Un mismatch entre el preprocesamiento de entrenamiento y el de
evaluacion es el error mas caro posible en este proyecto: el agente aprende un
mundo y se le evalua en otro.

Nota critica: AtariVectorEnv trae repeat_action_probability=0.0 por defecto,
pero ALE/SpaceInvaders-v5 usa 0.25. EnvConfig fuerza 0.25 en ambos caminos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import ale_py
import gymnasium as gym
import numpy as np
from ale_py.vector_env import AtariVectorEnv

gym.register_envs(ale_py)


@dataclass(frozen=True)
class EnvConfig:
    """Configuracion del entorno. Se serializa junto a cada checkpoint."""

    env_id: str = "ALE/SpaceInvaders-v5"
    game: str = "space_invaders"

    # --- semantica del juego: replica los defaults de ALE/SpaceInvaders-v5 ---
    repeat_action_probability: float = 0.25
    frameskip: int = 4
    full_action_space: bool = False
    max_num_frames_per_episode: int = 108_000

    # --- preprocesamiento visual ---
    img_height: int = 84
    img_width: int = 84
    grayscale: bool = True
    stack_num: int = 4
    maxpool: bool = True
    noop_max: int = 30

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        channels = self.stack_num if self.grayscale else 3 * self.stack_num
        return (channels, self.img_height, self.img_width)

    @property
    def n_actions(self) -> int:
        return 18 if self.full_action_space else 6

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = EnvConfig()

ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]


def make_vector_env(
    cfg: EnvConfig = DEFAULT_CONFIG,
    *,
    num_envs: int,
    episodic_life: bool,
    reward_clipping: bool,
    num_threads: int = 0,
) -> AtariVectorEnv:
    """Entorno vectorizado nativo (C++). Camino de entrenamiento y de evaluacion masiva.

    episodic_life/reward_clipping se activan para entrenar y se apagan para medir
    score real.
    """
    return AtariVectorEnv(
        game=cfg.game,
        num_envs=num_envs,
        repeat_action_probability=cfg.repeat_action_probability,
        full_action_space=cfg.full_action_space,
        max_num_frames_per_episode=cfg.max_num_frames_per_episode,
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        grayscale=cfg.grayscale,
        stack_num=cfg.stack_num,
        frameskip=cfg.frameskip,
        maxpool=cfg.maxpool,
        noop_max=cfg.noop_max,
        episodic_life=episodic_life,
        reward_clipping=reward_clipping,
        # Space Invaders arranca solo; el fire-reset nativo de ALE no es
        # replicable exactamente en el camino gym.make y rompia la equivalencia.
        use_fire_reset=False,
        num_threads=num_threads,
    )


def make_single_env(
    cfg: EnvConfig = DEFAULT_CONFIG,
    *,
    render_mode: str | None = None,
    video_folder: str | None = None,
    name_prefix: str = "agente",
) -> gym.Env:
    """Entorno unico via gym.make(ALE/SpaceInvaders-v5), con el mismo preprocesamiento.

    Este es el camino oficial de evaluacion: es literalmente el env_id que pide
    el enunciado. RecordVideo se inserta antes del preprocesamiento para grabar
    a resolucion completa y a 60 fps.
    """
    if video_folder is not None and render_mode is None:
        render_mode = "rgb_array"

    env = gym.make(
        cfg.env_id,
        frameskip=1,  # AtariPreprocessing aplica el frameskip
        repeat_action_probability=cfg.repeat_action_probability,
        full_action_space=cfg.full_action_space,
        max_num_frames_per_episode=cfg.max_num_frames_per_episode,
        render_mode=render_mode,
    )

    if video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda _: True,
            name_prefix=name_prefix,
        )

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=cfg.noop_max,
        frame_skip=cfg.frameskip,
        screen_size=(cfg.img_width, cfg.img_height),
        terminal_on_life_loss=False,
        grayscale_obs=cfg.grayscale,
        scale_obs=False,
    )

    # padding_type="zero" replica como AtariVectorEnv rellena el stack al
    # reiniciar; con el default ("reset") las dos rutas difieren 3 steps.
    env = gym.wrappers.FrameStackObservation(env, cfg.stack_num, padding_type="zero")
    return env


def make_raw_env(
    cfg: EnvConfig = DEFAULT_CONFIG,
    *,
    render_mode: str | None = None,
    video_folder: str | None = None,
    name_prefix: str = "agente",
) -> gym.Env:
    """Entorno sin preprocesamiento (RGB 210x160x3), para baselines de pixeles crudos."""
    if video_folder is not None and render_mode is None:
        render_mode = "rgb_array"

    env = gym.make(
        cfg.env_id,
        frameskip=cfg.frameskip,
        repeat_action_probability=cfg.repeat_action_probability,
        full_action_space=cfg.full_action_space,
        max_num_frames_per_episode=cfg.max_num_frames_per_episode,
        render_mode=render_mode,
    )

    if video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda _: True,
            name_prefix=name_prefix,
        )

    return env


def describe_env(cfg: EnvConfig = DEFAULT_CONFIG) -> dict:
    """Metadatos del entorno para el informe y para versionar checkpoints."""
    env = gym.make(cfg.env_id)
    unwrapped = env.unwrapped
    info = {
        "env_id": cfg.env_id,
        "gymnasium": gym.__version__,
        "ale_py": ale_py.__version__,
        "numpy": np.__version__,
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "action_meanings": unwrapped.get_action_meanings(),
        "spec_kwargs": dict(env.spec.kwargs),
        "sticky_action_prob": unwrapped.ale.getFloat("repeat_action_probability"),
        "preprocessed_obs_shape": cfg.obs_shape,
    }
    env.close()
    return info
