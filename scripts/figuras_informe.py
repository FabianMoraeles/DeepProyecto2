from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "runs"
REP = RAIZ / "reports"
INF = REP / "informe"
OUT = REP / "figures" / "informe"

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "axes.labelcolor": TINTA2,
    "axes.edgecolor": EJE,
    "axes.facecolor": SUPERFICIE,
    "figure.facecolor": SUPERFICIE,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TENUE,
    "ytick.color": TENUE,
    "xtick.labelcolor": TINTA2,
    "ytick.labelcolor": TINTA2,
    "legend.frameon": False,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "text.color": TINTA,
    "axes.titlecolor": TINTA,
})

NOMBRES = {
    "E01_dqn_seed0": "2. DQN base",
    "E02_double_seed0": "3. Double DQN",
    "E03_dueling_seed0": "4. Dueling",
    "E04_nstep_seed0": "5. Retornos de 3 pasos",
    "E05_per_seed0": "6. Repetición priorizada",
    "E06_iqn_seed0": "7. IQN",
    "E07_champion_seed0": "8. Configuración 5 prolongada",
    "E08_hreward_seed0": "9. Sin recorte de recompensa",
    "E09_gamma997_seed0": "10. Descuento 0.997",
}
ABLACIONES = ["E01_dqn_seed0", "E02_double_seed0", "E03_dueling_seed0",
              "E04_nstep_seed0", "E05_per_seed0"]

_cache: dict[str, EventAccumulator] = {}


def serie(run: str, tag: str, hasta: float | None = None):
    if run not in _cache:
        acc = EventAccumulator(str(RUNS / run / "tensorboard"), size_guidance={"scalars": 0})
        acc.Reload()
        _cache[run] = acc
    acc = _cache[run]
    if tag not in acc.Tags()["scalars"]:
        return np.array([]), np.array([])
    pts = acc.Scalars(tag)
    x = np.array([p.step for p in pts], dtype=float)
    y = np.array([p.value for p in pts], dtype=float)
    if hasta is not None:
        m = x <= hasta
        x, y = x[m], y[m]
    return x, y


def millones(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v / 1e6:g}"))
    ax.set_xlabel("Pasos de entorno, millones")


def guardar(fig, n: int):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"figura_{n:02d}.png", dpi=200, facecolor=SUPERFICIE)
    plt.close(fig)
    print(f"  figura {n:2d}")


def etiqueta_final(ax, x, y, texto, color, dy=0):
    ax.annotate(texto, xy=(x[-1], y[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=10.5, color=TINTA2)
    ax.plot([x[-1]], [y[-1]], "o", color=color, ms=5, mec=SUPERFICIE, mew=1.5)


def fotogramas_entorno():
    from si_rl.envs import DEFAULT_CONFIG, make_single_env
    from si_rl.policies import load_agent

    politica, cfg, _ = load_agent(RAIZ / "checkpoints" / "champion.pt", epsilon=0.0)
    env = make_single_env(cfg)
    obs, _ = env.reset(seed=4242)
    for _ in range(140):
        obs, *_ = env.step(politica(obs))
    crudo = env.unwrapped.ale.getScreenRGB().copy()
    pila = np.asarray(obs).copy()
    env.close()
    return crudo, pila


def fig01_02():
    crudo, pila = fotogramas_entorno()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5), gridspec_kw={"width_ratios": [160, 150]})
    axs[0].imshow(crudo, interpolation="nearest")
    axs[0].set_title("Original, 210 × 160 en color")
    axs[1].imshow(pila[-1], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    axs[1].set_title("Preprocesado, 84 × 84 en grises")
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    guardar(fig, 1)

    fig, axs = plt.subplots(1, 4, figsize=(8, 2.6))
    for k, a in enumerate(axs):
        a.imshow(pila[k], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        a.set_title(["t − 3", "t − 2", "t − 1", "t"][k])
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    guardar(fig, 2)


def fig03():
    d = json.loads((INF / "senal_aleatoria.json").read_text(encoding="utf-8"))
    valores = {int(k): v for k, v in d["valores"].items()}
    xs = sorted(valores)
    fig, ax = plt.subplots(figsize=(8, 4))
    barras = ax.bar([str(v) for v in xs], [valores[v] for v in xs], color=C[0], width=0.62)
    barras[-1].set_color(C[1])
    for b, v in zip(barras, xs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, str(valores[v]),
                ha="center", va="bottom", fontsize=10.5, color=TINTA2)
    ax.set_xlabel("Valor de la recompensa en puntos")
    ax.set_ylabel(f"Eventos en {d['episodios']} episodios")
    ax.text(len(xs) - 1, valores[200] + 14, "Nave nodriza", ha="center", fontsize=10.5,
            color=TINTA2)
    ax.grid(axis="x", visible=False)
    guardar(fig, 3)


def fig04():
    a = json.loads((INF / "senal_aleatoria.json").read_text(encoding="utf-8"))
    c = json.loads((INF / "comportamiento_campeon.json").read_text(encoding="utf-8"))
    bins = np.arange(0, 161, 5)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.clip(a["intervalos"], 0, 160), bins=bins, density=True, color=C[0],
            alpha=0.85, label=f"Aleatorio, {100 * a['fraccion_pasos_con_recompensa']:.2f} "
                              "por ciento de pasos con recompensa",
            edgecolor=SUPERFICIE, linewidth=1)
    ax.hist(np.clip(c["intervalos"], 0, 160), bins=bins, density=True, color=C[1],
            alpha=0.75, label=f"Agente final, {100 * c['fraccion_pasos_con_recompensa']:.2f} "
                              "por ciento de pasos con recompensa",
            edgecolor=SUPERFICIE, linewidth=1)
    ax.set_xlabel("Pasos entre dos recompensas consecutivas")
    ax.set_ylabel("Densidad")
    ax.legend(loc="upper right")
    guardar(fig, 4)


def fig05():
    ale = json.loads((REP / "etapa0_baseline_aleatorio.json").read_text(encoding="utf-8"))
    reg = json.loads((REP / "etapa0_baseline_regla_simple.json").read_text(encoding="utf-8"))
    datos = [ale["scores"], reg["scores"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    rng = np.random.default_rng(0)
    for i, (d, col) in enumerate(zip(datos, C[:2])):
        ax.boxplot([d], positions=[i], widths=0.45, showfliers=False,
                   medianprops={"color": TINTA, "linewidth": 2},
                   boxprops={"color": EJE}, whiskerprops={"color": EJE},
                   capprops={"color": EJE})
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=34, color=col,
                   edgecolor=SUPERFICIE, linewidth=1, zorder=3)
        ax.text(i + 0.3, np.mean(d), f"media {np.mean(d):.1f}", va="center",
                fontsize=10.5, color=TINTA2)
    ax.set_xticks([0, 1], ["Política aleatoria", "Regla por píxeles"])
    ax.set_ylabel("Puntaje por episodio")
    ax.set_xlim(-0.6, 1.9)
    ax.grid(axis="x", visible=False)
    guardar(fig, 5)


def fig06():
    d = json.loads((REP / "etapa0_check_env.json").read_text(encoding="utf-8"))
    tp = {int(k): v for k, v in d["throughput_steps_por_segundo"].items()}
    xs = [1] + sorted(tp)
    ys = [3288] + [tp[k] for k in sorted(tp)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, "-o", color=C[0], ms=8, mec=SUPERFICIE, mew=1.5)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:,.0f}".replace(",", " "), (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=10.5, color=TINTA2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("Entornos en paralelo")
    ax.set_ylabel("Pasos de agente por segundo")
    ax.set_ylim(0, 18000)
    guardar(fig, 6)


def fig07():
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    relleno, borde = "#cde2fb", "#86b6ef"

    def caja(x, y, w, h, texto, fc=relleno, ec=borde):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=10.2,
                color=TINTA, linespacing=1.35)

    def flecha(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=TENUE, lw=1.3))

    y = 38; h = 15
    caja(1, y, 14, h, "Entrada\n4 × 84 × 84")
    caja(20, y, 16, h, "Convolución\n32 filtros 8 × 8\npaso 4, 20 × 20")
    caja(41, y, 16, h, "Convolución\n64 filtros 4 × 4\npaso 2, 9 × 9")
    caja(62, y, 16, h, "Convolución\n64 filtros 3 × 3\npaso 1, 7 × 7")
    caja(83, y, 16, h, "Aplanado\n3136")
    for x0, x1 in [(15.8, 19.2), (36.8, 40.2), (57.8, 61.2), (78.8, 82.2)]:
        flecha(x0, y + h / 2, x1, y + h / 2)

    caja(52, 12, 20, 13, "Valor del estado\n512 unidades → 1", fc="#d2f0e4", ec="#1baf7a")
    caja(76, 12, 22, 13, "Ventaja por acción\n512 unidades → 6", fc="#fbe1d6", ec="#eb6834")
    flecha(91, y - 0.8, 87, 25.8)
    flecha(88, y - 0.8, 63, 25.8)
    caja(8, 12, 36, 13, "Q = V + A − media de A\nun valor por cada una\nde las 6 acciones",
         fc="#f0efec", ec=EJE)
    flecha(51.2, 18.5, 44.8, 18.5)
    ax.add_patch(FancyArrowPatch((87, 11.2), (30, 11.2), arrowstyle="-|>",
                                 connectionstyle="arc3,rad=-0.18", mutation_scale=12,
                                 color=TENUE, lw=1.3))
    ax.text(50, 58, "Todas las capas ocultas usan activación ReLU", ha="center",
            fontsize=10.5, color=TINTA2)
    guardar(fig, 7)


def fig08():
    pasos = np.linspace(0, 2_000_000, 400)
    eps = np.clip(1.0 + (0.01 - 1.0) * pasos / 1_000_000, 0.01, 1.0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axvspan(0, 80_000, color="#f0efec", zorder=0)
    ax.plot(pasos, eps, color=C[0])
    ax.text(90_000, 0.93, "Acciones aleatorias mientras\nse llena la memoria", fontsize=10.5,
            color=TINTA2, va="top")
    ax.annotate("0.01 a partir de 1 millón de pasos", xy=(1_000_000, 0.01),
                xytext=(1_150_000, 0.28), fontsize=10.5, color=TINTA2,
                arrowprops={"arrowstyle": "-", "color": TENUE, "lw": 1})
    millones(ax)
    ax.set_ylabel("Épsilon")
    ax.set_ylim(0, 1.05)
    guardar(fig, 8)


def fig09():
    comp = ["Paso del entorno", "Muestreo de la memoria", "Transferencia a la GPU",
            "Actualización de la red", "Selección de acción"]
    ms = [4.28, 13.81, 1.46, 4.11, 0.61]
    total = sum(ms)
    y = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    colores = [C[1] if i == 1 else C[0] for i in range(len(comp))]
    ax.barh(y, ms, height=0.58, color=colores)
    for yi, v in zip(y, ms):
        ax.text(v + 0.2, yi, f"{v:.2f} ms, {100 * v / total:.0f} por ciento", va="center",
                fontsize=10.5, color=TINTA2)
    ax.set_yticks(y, comp)
    ax.invert_yaxis()
    ax.set_xlabel(f"Milisegundos por paso de 64 transiciones, total {total:.1f}")
    ax.set_xlim(0, 20)
    ax.grid(axis="y", visible=False)
    guardar(fig, 9)


def curvas(runs, tag, n, hasta=None, ylabel="Puntaje", etiquetas=True, figsize=(8, 3.9),
           dy=None):
    fig, ax = plt.subplots(figsize=figsize)
    for i, r in enumerate(runs):
        x, y = serie(r, tag, hasta)
        if len(x):
            ax.plot(x, y, color=C[i], label=NOMBRES[r])
    millones(ax)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=10.5)
    guardar(fig, n)


def fig10_11_12():
    curvas(ABLACIONES, "eval/score_medio", 10, hasta=4_000_000,
           ylabel="Puntaje medio de evaluación")
    curvas(ABLACIONES, "episodio/score_medio_200", 11, hasta=4_000_000,
           ylabel="Media móvil de 200 episodios")
    curvas(ABLACIONES, "entrenamiento/q_medio", 12, hasta=4_000_000,
           ylabel="Valor Q medio del lote")


def fig13():
    df = pd.read_csv(REP / "comparacion_step_4000000.csv")
    orden = ["E01_dqn_seed0", "E02_double_seed0", "E03_dueling_seed0", "E04_nstep_seed0",
             "E05_per_seed0", "E06_iqn_seed0"]
    corto = ["DQN base", "Double", "Dueling", "3 pasos", "Priorizada", "IQN"]
    df = df.set_index("experimento").loc[orden]
    x = np.arange(len(orden))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, df["media"], width=0.38, color=C[0], label="Media")
    ax.bar(x + 0.2, df["E_max_de_5"], width=0.38, color=C[1],
           label="Máximo esperado de 5 episodios")
    ax.set_xticks(x, corto)
    ax.set_ylabel("Puntaje, 100 episodios")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    guardar(fig, 13)


def fig14():
    a = json.loads((REP / "sweep_epsilon_final.json").read_text(encoding="utf-8"))["resultados"]
    b = json.loads((REP / "sweep_epsilon_mejor.json").read_text(encoding="utf-8"))["resultados"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.9))
    for ax, d, tit in [(axs[0], a, "Configuración 5, 4 M pasos"),
                       (axs[1], b, "Configuración 8, 50 M pasos")]:
        eps = [str(f["epsilon"]) for f in d]
        ax.plot(eps, [f["E_max_de_5"] for f in d], "-o", color=C[1], ms=7,
                mec=SUPERFICIE, mew=1.5, label="Máximo esperado de 5")
        ax.plot(eps, [f["media"] for f in d], "-o", color=C[0], ms=7,
                mec=SUPERFICIE, mew=1.5, label="Media")
        ax.set_title(tit)
        ax.set_xlabel("Épsilon de evaluación")
    axs[0].set_ylabel("Puntaje")
    axs[1].legend(loc="lower left", fontsize=10)
    guardar(fig, 14)


def fig15():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    runs = ["E01_dqn_seed0", "E04_nstep_seed0", "E06_iqn_seed0"]
    cols = [C[0], C[3], C[5]]
    for r, col in zip(runs, cols):
        x, y = serie(r, "eval/score_medio")
        ax.plot(x, y, color=col, label=NOMBRES[r])
    millones(ax)
    ax.set_ylabel("Puntaje medio de evaluación")
    ax.legend(loc="upper left")
    guardar(fig, 15)


def fig16():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x, y = serie("E07_champion_seed0", "episodio/score_medio_200")
    ax.plot(x, y, color=C[0], label="Entrenamiento, media móvil de 200 episodios")
    x, y = serie("E07_champion_seed0", "eval/score_medio")
    ax.plot(x, y, color=C[1], label="Evaluación, 50 episodios")
    millones(ax)
    ax.set_ylabel("Puntaje")
    ax.legend(loc="lower right")
    guardar(fig, 16)


def fig17():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    runs = ["E07_champion_seed0", "E08_hreward_seed0", "E09_gamma997_seed0"]
    for i, r in enumerate(runs):
        x, y = serie(r, "eval/score_medio", 20_000_000)
        ax.plot(x, y, color=C[i], label=NOMBRES[r])
    millones(ax)
    ax.set_ylabel("Puntaje medio de evaluación")
    ax.legend(loc="upper left")
    guardar(fig, 17)


def fig18():
    df = pd.read_csv(REP / "seleccion_checkpoint_E07_champion_seed0.csv")
    pasos = df[df["checkpoint"].str.startswith("step_")].sort_values("steps")
    mejor = df[df["checkpoint"] == "mejor.pt"].iloc[0]
    se = pasos["desv"] / np.sqrt(200)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.errorbar(pasos["steps"], pasos["media"], yerr=1.96 * se, fmt="-o", color=C[0],
                ms=6, mec=SUPERFICIE, capsize=3, label="Media con intervalo de 95 por ciento")
    ax.plot(pasos["steps"], pasos["E_max_de_5"], "-o", color=C[1], ms=6, mec=SUPERFICIE,
            label="Máximo esperado de 5")
    ax.plot([mejor["steps"]], [mejor["media"]], "D", color=TINTA, ms=8, zorder=5)
    ax.annotate("Punto elegido durante\nel entrenamiento", (mejor["steps"], mejor["media"]),
                xytext=(-150, -38), textcoords="offset points", fontsize=10.5, color=TINTA2,
                arrowprops={"arrowstyle": "-", "color": TENUE, "lw": 1})
    millones(ax)
    ax.set_ylabel("Puntaje, 200 episodios")
    ax.set_ylim(1800, 3200)
    ax.legend(loc="lower right", fontsize=10.5)
    guardar(fig, 18)


def fig19():
    df = pd.read_csv(RAIZ / "data" / "evaluation" / "leaderboard_champion.csv")
    frac = (df["score"] == 2815).mean() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["score"], bins=np.arange(0, 3601, 100), color=C[0], edgecolor=SUPERFICIE,
            linewidth=1.2)
    ax.axvline(2815, color=C[1], lw=2)
    ax.text(2780, ax.get_ylim()[1] * 0.92,
            f"2815 puntos:\n{frac:.0f} por ciento\nde los episodios", ha="right", va="top",
            fontsize=10.5, color=TINTA2)
    ax.set_xlabel("Puntaje por episodio")
    ax.set_ylabel(f"Episodios de {len(df)}")
    guardar(fig, 19)
    return frac


def fig20():
    df = pd.read_csv(RAIZ / "data" / "evaluation" / "leaderboard_champion.csv")
    r = df["score"].corr(df["steps"])
    c = json.loads((INF / "comportamiento_campeon.json").read_text(encoding="utf-8"))
    por_ep = {}
    for p in c["vidas_perdidas"]:
        por_ep.setdefault(p["episodio"], []).append(p)
    abatido, invasion = [], []
    for f in c["finales"]:
        evs = por_ep[f["episodio"]]
        previas = 3 if len(evs) == 1 else evs[-2]["vidas_restantes"]
        (abatido if previas - evs[-1]["vidas_restantes"] == 1 else invasion).append(f["puntaje"])

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].scatter(df["steps"], df["score"], s=22, color=C[0], edgecolor=SUPERFICIE,
                   linewidth=0.8, alpha=0.9)
    axs[0].set_xlabel("Duración del episodio en pasos")
    axs[0].set_ylabel("Puntaje")
    axs[0].set_title(f"Correlación {r:.2f}, {len(df)} episodios")

    rng = np.random.default_rng(1)
    for i, (d, col) in enumerate([(abatido, C[0]), (invasion, C[1])]):
        axs[1].scatter(i + rng.uniform(-0.13, 0.13, len(d)), d, s=30, color=col,
                       edgecolor=SUPERFICIE, linewidth=0.8)
    axs[1].set_xticks([0, 1], [f"Abatido\n{len(abatido)} episodios",
                               f"Invasión\n{len(invasion)} episodios"])
    axs[1].set_xlim(-0.5, 1.5)
    axs[1].set_title("Forma de terminar")
    axs[1].grid(axis="x", visible=False)
    guardar(fig, 20)
    return r, len(abatido), len(invasion), abatido, invasion


def fig21():
    video = RAIZ / "reports" / "videos" / "record_champion" / "record_champion-episode-0.mp4"
    frames = iio.imread(video, plugin="pyav")
    n = len(frames)
    idx = [int(n * f) for f in (0.08, 0.40, 0.72, 0.995)]
    fig, axs = plt.subplots(1, 4, figsize=(8, 3.4))
    for a, i, t in zip(axs, idx, ["Inicio", "Primer tercio", "Segundo tercio", "Final"]):
        a.imshow(frames[i], interpolation="nearest")
        a.set_title(t)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    guardar(fig, 21)
    return n


def fig22():
    c = json.loads((INF / "comportamiento_campeon.json").read_text(encoding="utf-8"))
    nombres = ["Nada", "Disparar", "Derecha", "Izquierda", "Derecha\ny\ndisparar",
               "Izquierda\ny\ndisparar"]
    tot = sum(c["acciones"].values())
    frac = [100 * c["acciones"].get(str(k), c["acciones"].get(k, 0)) / tot for k in range(6)]
    x = np.array(c["posiciones"])
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.3), gridspec_kw={"width_ratios": [1.7, 1]})
    b = axs[0].bar(range(6), frac, color=C[0], width=0.62)
    for bi, f in zip(b, frac):
        axs[0].text(bi.get_x() + bi.get_width() / 2, f + 0.8, f"{f:.1f}", ha="center",
                    fontsize=10, color=TINTA2)
    axs[0].set_xticks(range(6), nombres, fontsize=9.5)
    axs[0].set_ylabel("Por ciento de las acciones")
    axs[0].grid(axis="x", visible=False)
    axs[1].hist(x, bins=np.arange(x.min(), x.max() + 2, 3), color=C[1],
                edgecolor=SUPERFICIE, linewidth=0.8, density=True)
    axs[1].set_xlabel("Columna del cañón en píxeles")
    axs[1].set_ylabel("Densidad")
    guardar(fig, 22)
    return frac


def main() -> None:
    fig01_02(); fig03(); fig04(); fig05(); fig06(); fig07(); fig08(); fig09()
    fig10_11_12(); fig13(); fig14(); fig15(); fig16(); fig17(); fig18()
    frac = fig19()
    r, na, ni, ab, inv = fig20()
    nfr = fig21()
    acc = fig22()
    resumen = {"fraccion_2815": frac, "corr_puntaje_duracion": r, "abatido": na,
               "invasion": ni, "puntajes_abatido": ab, "puntajes_invasion": inv,
               "fotogramas_video": nfr, "acciones_por_ciento": acc}
    (INF / "resumen_figuras.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumen.items() if "puntajes" not in k}, indent=2))


if __name__ == "__main__":
    main()
