# Space Invaders RL — Proyecto 2

Agente de Reinforcement Learning para `ALE/SpaceInvaders-v5`.
CC3092 Deep Learning y Sistemas Inteligentes.

## Instalación

```bash
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu129
uv pip install -e .
```

Requiere GPU NVIDIA con CUDA 12.8+ (probado en RTX 5070 Ti, compute capability 12.0).

## Verificar la instalación

```bash
python scripts/check_env.py     # versiones, GPU, entorno, equivalencia de rutas, throughput
python -m pytest tests/ -q      # 15 tests
```

## Evaluar un modelo entrenado

```bash
python scripts/evaluate.py runs/E01_dqn_seed0/mejor.pt
```

Corre 5 episodios con política greedy, reporta el máximo (métrica de la
competencia) y graba el video de la mejor corrida en `reports/videos/`.

El preprocesamiento **no se configura al evaluar**: viaja dentro del checkpoint
junto con los pesos, así que es imposible evaluar con wrappers distintos a los
del entrenamiento.

## Entrenar

```bash
python scripts/train.py --smoke                              # validación rápida (~1 min)
python scripts/train.py --exp-id E01_dqn --total-steps 10000000
tensorboard --logdir runs/
```

Para **continuar** una corrida desde donde quedó:

```bash
python scripts/train.py --exp-id E07_champion --resume runs/E07_champion_seed0/final.pt \
    --total-steps 80000000 --double-dqn true --dueling true --n-step 3
```

El replay buffer no se guarda (pesa varios GB), así que la corrida vuelve a
llenarlo antes de entrenar; los pesos, el estado de Adam y el contador de pasos
sí continúan. En la práctica el agente recupera su nivel previo en ~100k pasos.

## Arquitectura del entorno

Hay dos rutas de construcción de entorno y **se verificó que producen
observaciones bit-idénticas** (`tests/test_environment.py::test_rutas_equivalentes`):

| Ruta | Uso | Por qué |
|---|---|---|
| `make_vector_env` | entrenamiento | `AtariVectorEnv` nativo en C++, ~15k steps/s con 64 entornos |
| `make_single_env` | evaluación y video | literalmente `gym.make("ALE/SpaceInvaders-v5")`, que es lo que pide el enunciado |

Dos diferencias entre ambas implementaciones tuvieron que corregirse para que
coincidieran:

- **`use_fire_reset`**: ALE lo implementa de forma no replicable en la ruta
  `gym.make`. Space Invaders arranca solo, así que se desactiva en ambas.
- **Padding del frame stack al reiniciar**: `AtariVectorEnv` rellena con ceros,
  `FrameStackObservation` repite el primer frame. Se fuerza `padding_type="zero"`.

`AtariVectorEnv` además trae `repeat_action_probability=0.0` por defecto,
mientras que `ALE/SpaceInvaders-v5` usa **0.25**. `EnvConfig` fuerza 0.25 en
ambas rutas; entrenar con el default habría producido un agente que colapsa el
día de la evaluación.

## Estructura

```
src/si_rl/
    envs.py        EnvConfig y fábricas de entornos (fuente única de verdad)
    networks.py    CNN de Nature, cabeza dueling, IQN
    agents.py      DQNAgent e IQNAgent (acción + pérdida)
    replay.py      replay uniforme y priorizado, con deduplicación de frames
    policies.py    política greedy, guardado y carga de checkpoints
    evaluate.py    harness de evaluación (score real, E[max de 5])
    baselines.py   políticas aleatoria y de regla simple
    train_dqn.py   bucle de entrenamiento (agnóstico al algoritmo)
scripts/
    check_env.py           diagnóstico del stack completo
    eval_baselines.py      baselines de referencia
    train.py               CLI de entrenamiento
    run_ablations.py       cadena de ablaciones aditivas
    evaluate.py            evaluación + video (script de la presentación)
    compare_experiments.py torneo interno con protocolo idéntico
    sweep_eval_epsilon.py  barrido de epsilon contra la métrica max-de-5
    sweep_risk.py          barrido de distorsiones de riesgo (solo IQN)
    hunt_high_score.py     búsqueda del récord y grabación del MP4 final
    freeze_champion.py     congela el campeón con trazabilidad completa
    plot_curves.py         curvas de entrenamiento para el informe
    monitor.py             estado de una corrida en curso
tests/             45 tests
runs/              checkpoints, TensorBoard y métricas por experimento
reports/           resultados, figuras y videos
```

## Algoritmos

El bucle de entrenamiento es agnóstico al algoritmo: entorno, replay, logging,
checkpoints y evaluación viven una sola vez en `train_dqn.py`, y lo único que
cambia por algoritmo está en `agents.py`. Eso garantiza que las ablaciones se
comparen bajo exactamente el mismo bucle.

| Flag | Efecto |
|---|---|
| `--algo dqn\|iqn` | DQN clásico o Implicit Quantile Network |
| `--double-dqn true` | la red en línea elige la acción, la objetivo la valúa |
| `--dueling true` | `Q = V + A − mean(A)` |
| `--n-step 3` | retornos n-step |
| `--per true` | replay priorizado con pesos de importancia |

**Por qué IQN.** En vez de estimar el valor esperado `Q(s,a)`, estima la función
cuantil de la distribución de retornos. Space Invaders es el juego donde más se
nota: a 200M frames los números publicados son ~1,976 para DQN, ~5,699 para
Rainbow y ~28,888 para IQN. Además permite aplicar distorsiones *risk-seeking*
al evaluar, lo que ataca directamente la métrica de máximo-de-5.

## Decisiones de diseño

**Replay buffer con deduplicación de frames.** Guardar la observación apilada
(4, 84, 84) por transición desperdicia 4x memoria: dos pasos consecutivos del
mismo entorno comparten 3 frames. 1M de transiciones costarían 26 GB apiladas
contra 6.6 GB deduplicadas. La reconstrucción se apoya en una invariante de
escritura en lockstep y está verificada contra el entorno real en
`tests/test_replay.py::test_stack_reconstruido_coincide_con_el_entorno`.

**`episodic_life` desactivado.** ALE no expone si una terminación fue pérdida de
vida o fin de partida, lo que impide distinguirlas para el bootstrap y para
registrar el score. Las vidas se llevan a mano con `info['lives']`.

**Reward clipping aplicado fuera del entorno.** El entorno entrega el score
real (que es lo que se registra y se reporta) y el entrenamiento usa
`np.sign(r)`. Así nunca se confunde la señal de entrenamiento con la métrica.

**Sin AMP.** La CNN de Nature es demasiado pequeña para que los tensor cores
compensen el costo de `autocast`: medido, bf16 tarda 4.11 ms por update contra
3.39 ms en fp32 con TF32.

## Métrica

La competencia toma el **máximo de 5 episodios**, no la media. El harness
reporta media, mediana y desviación estándar para diagnóstico, pero también
estima `E[max de 5]` por bootstrap, que es lo que realmente predice el
desempeño el día de la presentación.

Eso cambia qué política conviene al evaluar: la neutral al riesgo maximiza el
retorno *esperado*, que no es lo que se califica. Dos palancas, ambas sin
reentrenar:

- `sweep_eval_epsilon.py` — algo de exploración residual baja la media pero
  puede engordar la cola superior.
- `sweep_risk.py` — solo para IQN. Como aprende la distribución completa de
  retornos, basta cambiar cómo se muestrean los cuantiles al evaluar:
  `cvar_sup` concentra la valuación en la cola superior, que es justamente lo
  que se puntúa.

## Hallazgos experimentales

Tres intentos de mejora **fallaron**, y los tres fallan por la misma razón. Vale
la pena documentarlos porque acotan el espacio de diseño de este juego.

| Cambio | Resultado @20M | vs. línea base |
|---|---|---|
| Línea base (clip, γ=0.99) | 2024 media / 2865 E[max5] | — |
| Quitar reward clipping (reescalado h) | 981 / 1651 | **−52%** |
| γ = 0.997 (horizonte más largo) | 1816 / 2545 | −10% |
| PER | ver fase 2 | −17% |
| Evaluación *risk-seeking* (IQN) | ver `sweep_risk` | −12% |

**El patrón: todo lo que incentiva la codicia hace daño.** En Space Invaders
morir termina el episodio, y el score está correlacionado 0.91 con la duración —
es un juego de supervivencia, no de puntería. Quitar el clipping le enseña al
agente que la nave nodriza vale 40x un alien, pero cazarla obliga a salir de
cobertura: el incentivo se paga con la vida. Las distorsiones *risk-seeking*
fallan por lo mismo. Visto así, el reward clipping no es "pérdida de
información" sino que aplana justamente la tentación de perseguir premios
grandes.

**Selección de checkpoint.** `mejor.pt` se elige durante el entrenamiento con
evaluaciones de 50 episodios sobre una distribución de desviación ~800: el error
estándar es ~115 y tomar el máximo sobre decenas de evaluaciones selecciona la
evaluación afortunada. Re-evaluando 14 checkpoints con 200 episodios
(`select_checkpoint.py`), `mejor.pt` quedó **11º de 14**. Además `E[max de 5]`
resultó plano entre todos ellos (2887–2964, dentro del ruido), así que el
desempate real fue media alta y varianza baja.

## Flujo de la entrega final

```bash
python scripts/compare_experiments.py --checkpoint final.pt --episodios 100  # torneo
python scripts/sweep_risk.py runs/<ganador>/mejor.pt                         # política de eval
python scripts/freeze_champion.py runs/<ganador>/mejor.pt --distorsion cvar_sup --eta 0.25
python scripts/hunt_high_score.py checkpoints/champion.pt --intentos 300     # MP4 récord
python scripts/evaluate.py checkpoints/champion.pt                           # día de la presentación
```
