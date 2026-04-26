"""
streamlit_app.py
----------------
Application Streamlit — thème "Mission Control" pour le projet
"Gestion intelligente du trafic urbain".

Identité visuelle :
- Fond navy profond (#0B1929), accents cyan (#00D9FF) + ambre (#FFA500)
- Carrefour façon centre de contrôle (halos lumineux sur les feux, voitures
  stylisées en barres lumineuses).
- KPIs en gros chiffres cyan, séparation visuelle par cartes.

Trois pages :
  1. "Simulation live"     : visualisation pas-à-pas du carrefour
  2. "Comparaison"         : comparaison côte-à-côte des politiques
  3. "Apprentissage"       : courbes de convergence du Q-Learning
"""

import os
import sys
import time

# Permet l'import de src/* depuis ce sous-dossier
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.agent import QLearningAgent
from src.baseline import FixedTimePolicy, LongestQueuePolicy
from src.environment import (
    ACTION_KEEP,
    ACTION_SWITCH,
    PHASE_NS,
    SCENARIOS,
    TrafficConfig,
    TrafficEnv,
)
from src.utils import load_metrics, load_q_table

RESULTS_DIR = os.path.join(_ROOT, "results")

# =================================================== Palette Mission Control
NAVY      = "#0B1929"
NAVY_2    = "#102841"
NAVY_3    = "#1A3A5C"
WHITE     = "#E8F0FE"
GRID      = "#1F3F5F"
CYAN      = "#00D9FF"
AMBER     = "#FFA500"
GREEN     = "#10F0A0"
RED       = "#FF4757"
PURPLE    = "#9D5CFF"
TEAL      = "#1ABC9C"

# Configure matplotlib pour le mode sombre par défaut
plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor": NAVY,
    "axes.edgecolor": NAVY_3,
    "axes.labelcolor": WHITE,
    "xtick.color": WHITE,
    "ytick.color": WHITE,
    "text.color": WHITE,
    "axes.titlecolor": WHITE,
    "grid.color": GRID,
    "grid.alpha": 0.4,
    "legend.facecolor": NAVY_2,
    "legend.edgecolor": NAVY_3,
    "legend.labelcolor": WHITE,
    "savefig.facecolor": NAVY,
})


# ===================================================================== CSS

CUSTOM_CSS = f"""
<style>
/* --- Typographie globale ----------------------------------------------- */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    letter-spacing: 0.01em;
}}

/* --- Header principal -------------------------------------------------- */
.mc-header {{
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    margin-bottom: 24px;
    background: linear-gradient(90deg, {NAVY_2} 0%, {NAVY} 100%);
    border-left: 4px solid {CYAN};
    border-radius: 6px;
    box-shadow: 0 0 24px rgba(0, 217, 255, 0.08);
}}
.mc-header h1 {{
    color: {WHITE};
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
}}
.mc-header .mc-sub {{
    color: {CYAN};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600;
    margin-top: 2px;
}}
.mc-pulse {{
    width: 12px; height: 12px; border-radius: 50%;
    background: {GREEN};
    box-shadow: 0 0 12px {GREEN};
    animation: pulse 1.6s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.6; transform: scale(1.2); }}
}}

/* --- Cartes KPI -------------------------------------------------------- */
.mc-card {{
    background: {NAVY_2};
    border: 1px solid {NAVY_3};
    border-radius: 8px;
    padding: 16px 20px;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.mc-card:hover {{
    border-color: {CYAN};
    box-shadow: 0 0 16px rgba(0, 217, 255, 0.15);
}}
.mc-kpi-label {{
    color: #7E97B5;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 6px;
}}
.mc-kpi-value {{
    color: {CYAN};
    font-size: 2rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    line-height: 1;
}}
.mc-kpi-value.amber {{ color: {AMBER}; }}
.mc-kpi-value.green {{ color: {GREEN}; }}
.mc-kpi-value.red   {{ color: {RED}; }}
.mc-kpi-unit {{
    color: #7E97B5;
    font-size: 0.85rem;
    margin-left: 4px;
}}

/* --- Sidebar ----------------------------------------------------------- */
section[data-testid="stSidebar"] {{
    background: {NAVY_2};
    border-right: 1px solid {NAVY_3};
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {CYAN};
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.85rem;
    font-weight: 700;
}}

/* --- Boutons ----------------------------------------------------------- */
.stButton > button {{
    background: linear-gradient(90deg, {CYAN} 0%, #00B8D9 100%);
    color: {NAVY};
    border: none;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.82rem;
    padding: 10px 24px;
    border-radius: 4px;
    transition: all 0.18s;
}}
.stButton > button:hover {{
    box-shadow: 0 0 18px {CYAN};
    transform: translateY(-1px);
}}

/* --- Live status bar (pendant la simulation) --------------------------- */
.mc-status {{
    background: {NAVY_2};
    border: 1px solid {NAVY_3};
    border-left: 3px solid {AMBER};
    padding: 10px 18px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 0.92rem;
    color: {WHITE};
}}
.mc-status .label {{ color: #7E97B5; }}
.mc-status .val   {{ color: {CYAN}; font-weight: 700; }}
.mc-status .val-amber {{ color: {AMBER}; font-weight: 700; }}
.mc-status .val-green {{ color: {GREEN}; font-weight: 700; }}
.mc-status .val-red   {{ color: {RED}; font-weight: 700; }}

/* --- Tableaux ---------------------------------------------------------- */
[data-testid="stTable"] {{
    background: {NAVY_2};
}}

/* --- Sliders ----------------------------------------------------------- */
[data-baseweb="slider"] [role="slider"] {{
    background: {CYAN} !important;
    box-shadow: 0 0 8px {CYAN} !important;
}}

/* --- Sélecteurs -------------------------------------------------------- */
[data-baseweb="select"] > div {{
    background: {NAVY_2} !important;
    border: 1px solid {NAVY_3} !important;
}}

/* --- Section titles ---------------------------------------------------- */
.mc-section {{
    color: {CYAN};
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-size: 0.85rem;
    font-weight: 700;
    margin: 28px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid {NAVY_3};
}}
</style>
"""


# ================================================================ Helpers

def render_header(title: str, subtitle: str):
    st.markdown(f"""
    <div class="mc-header">
        <div class="mc-pulse"></div>
        <div>
            <div class="mc-sub">{subtitle}</div>
            <h1>{title}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value, unit: str = "", color: str = "cyan"):
    return f"""
    <div class="mc-card">
        <div class="mc-kpi-label">{label}</div>
        <div class="mc-kpi-value {color}">{value}<span class="mc-kpi-unit">{unit}</span></div>
    </div>
    """


def section_title(text: str):
    st.markdown(f'<div class="mc-section">{text}</div>', unsafe_allow_html=True)


def make_policy(name: str, scenario: str, fixed_period: int = 10):
    """Instancie la politique demandée."""
    if name == "Q-Learning":
        agent = QLearningAgent()
        q_path = os.path.join(RESULTS_DIR, f"q_table_{scenario}.pkl")
        if os.path.exists(q_path):
            for k, v in load_q_table(q_path).items():
                agent.Q[k] = v
        return agent, len(agent.Q)
    elif name == "FixedTime":
        return FixedTimePolicy(period=fixed_period), 0
    elif name == "LongestQueue":
        return LongestQueuePolicy(), 0
    raise ValueError(f"Politique inconnue : {name}")


def render_intersection(queues, phase, ax, step=None):
    """
    Dessine le carrefour façon centre de contrôle :
    - Routes : bandes navy plus claires sur fond ultra-sombre
    - Lignes pointillées de marquage routier
    - Feux : disques avec halo lumineux (vert pulsant / rouge sourd)
    - Voitures : barres lumineuses dégradées vers la couleur de la branche
    """
    ax.clear()
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(NAVY)

    # Routes (bandes plus claires que le fond)
    ax.add_patch(plt.Rectangle((-1.4, -7), 2.8, 14, color=NAVY_3, zorder=0))
    ax.add_patch(plt.Rectangle((-7, -1.4), 14, 2.8, color=NAVY_3, zorder=0))

    # Marquage routier (lignes pointillées blanches au centre)
    for y in np.arange(-6.5, -1.6, 0.8):
        ax.plot([0, 0], [y, y + 0.4], color=WHITE, lw=1.2, alpha=0.35, zorder=1)
    for y in np.arange(2.0, 6.6, 0.8):
        ax.plot([0, 0], [y, y + 0.4], color=WHITE, lw=1.2, alpha=0.35, zorder=1)
    for x in np.arange(-6.5, -1.6, 0.8):
        ax.plot([x, x + 0.4], [0, 0], color=WHITE, lw=1.2, alpha=0.35, zorder=1)
    for x in np.arange(2.0, 6.6, 0.8):
        ax.plot([x, x + 0.4], [0, 0], color=WHITE, lw=1.2, alpha=0.35, zorder=1)

    # Centre du carrefour
    ax.add_patch(plt.Rectangle((-1.4, -1.4), 2.8, 2.8, color=NAVY_2,
                               ec=CYAN, lw=1.2, zorder=2))

    # --- Feux avec halo lumineux ---
    color_NS = GREEN if phase == PHASE_NS else RED
    color_EO = RED if phase == PHASE_NS else GREEN

    def light(x, y, color):
        # halo
        for r, alpha in zip([0.65, 0.45, 0.30], [0.10, 0.18, 0.32]):
            ax.add_patch(plt.Circle((x, y), r, color=color, alpha=alpha, zorder=3))
        # disque central
        ax.add_patch(plt.Circle((x, y), 0.22, color=color,
                                ec=WHITE, lw=0.8, zorder=4))

    light(0, 1.7, color_NS)    # feu Nord
    light(0, -1.7, color_NS)   # feu Sud
    light(1.7, 0, color_EO)    # feu Est
    light(-1.7, 0, color_EO)   # feu Ouest

    # --- Voitures sous forme de barres lumineuses ---
    qN, qS, qE, qO = queues

    def draw_cars(x, y, dx, dy, n, color):
        for i in range(min(n, 8)):
            xc = x + dx * i
            yc = y + dy * i
            # halo de la voiture
            ax.add_patch(plt.Rectangle(
                (xc - 0.45, yc - 0.28), 0.9, 0.56,
                color=color, alpha=0.18, zorder=5,
            ))
            # corps
            ax.add_patch(plt.Rectangle(
                (xc - 0.36, yc - 0.20), 0.72, 0.40,
                color=color, ec=WHITE, lw=0.5, zorder=6,
            ))

    draw_cars(0, 2.2, 0, 0.62, qN, CYAN)        # Nord
    draw_cars(0, -2.2, 0, -0.62, qS, PURPLE)    # Sud
    draw_cars(2.2, 0, 0.62, 0, qE, AMBER)       # Est
    draw_cars(-2.2, 0, -0.62, 0, qO, TEAL)      # Ouest

    # --- Labels par branche ---
    label_kwargs = dict(ha="center", va="center", fontsize=11, fontweight="bold",
                        path_effects=[pe.withStroke(linewidth=3, foreground=NAVY)])
    ax.text(0, 6.5, f"N : {qN}", color=CYAN, **label_kwargs)
    ax.text(0, -6.5, f"S : {qS}", color=PURPLE, **label_kwargs)
    ax.text(6.5, 0, f"E : {qE}", color=AMBER, **label_kwargs)
    ax.text(-6.5, 0, f"O : {qO}", color=TEAL, **label_kwargs)

    # --- Bandeau titre ---
    phase_label = "AXE NS · VERT" if phase == PHASE_NS else "AXE EO · VERT"
    title = f"PHASE : {phase_label}"
    if step is not None:
        title = f"T={step:03d}   ·   {title}"
    ax.text(0, 7.4, title, color=CYAN, ha="center", va="center",
            fontsize=11, fontweight="bold", family="monospace")


# =============================================================== Pages

def page_live():
    section_title("Simulation temps réel")
    st.markdown(
        f"<p style='color:#7E97B5;'>Visualisation pas-à-pas du carrefour "
        f"sous la politique sélectionnée. Les feux passent au "
        f"<span style='color:{GREEN}'>vert</span> sur l'axe autorisé "
        f"et au <span style='color:{RED}'>rouge</span> sur l'axe bloqué.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        scenario = st.selectbox("Scénario", list(SCENARIOS.keys()), index=0)
    with col2:
        policy_name = st.selectbox("Politique",
                                   ["Q-Learning", "LongestQueue", "FixedTime"])
    with col3:
        n_steps = st.number_input("Pas à simuler", 20, 500, 200, step=20)
    with col4:
        speed = st.slider("Vitesse (pas/s)", 1.0, 30.0, 10.0)

    fixed_period = st.slider("Période FixedTime (si applicable)", 2, 30, 10)

    if st.button("DEMARRER LA SIMULATION", type="primary"):
        config = SCENARIOS[scenario]
        env = TrafficEnv(config, seed=int(time.time()) % 10_000)
        policy, _ = make_policy(policy_name, scenario, fixed_period=fixed_period)

        s = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()

        # Layout : intersection à gauche, KPIs à droite
        layout_cols = st.columns([2, 1])
        with layout_cols[0]:
            chart_placeholder = st.empty()
        with layout_cols[1]:
            kpi_placeholder = st.empty()

        status_placeholder = st.empty()

        history = {"step": [], "reward": [], "queue_total": []}
        cum_reward = 0.0

        fig, ax = plt.subplots(figsize=(6, 6), facecolor=NAVY)

        for t in range(n_steps):
            a = policy.choose_action(s, greedy=True)
            s_prime, r, done, info = env.step(a)
            cum_reward += r
            history["step"].append(t + 1)
            history["reward"].append(r)
            history["queue_total"].append(sum(info.queues))

            render_intersection(info.queues, info.phase, ax, step=t + 1)
            chart_placeholder.pyplot(fig, use_container_width=True)

            # KPIs colonne droite
            queue_color = "green" if sum(info.queues) < 5 else \
                          ("amber" if sum(info.queues) < 12 else "red")
            with kpi_placeholder.container():
                st.markdown(kpi_card("Pas écoulés", f"{t+1}",
                                     unit=f" / {n_steps}"), unsafe_allow_html=True)
                st.markdown(kpi_card("File totale", sum(info.queues),
                                     unit=" véh.", color=queue_color),
                            unsafe_allow_html=True)
                st.markdown(kpi_card("Récompense cumulée", f"{cum_reward:.1f}",
                                     color="amber"), unsafe_allow_html=True)
                st.markdown(kpi_card("Politique", policy_name), unsafe_allow_html=True)

            action_label = "CHANGER" if a == ACTION_SWITCH else "MAINTENIR"
            action_color = "val-amber" if a == ACTION_SWITCH else "val-green"
            status_placeholder.markdown(
                f"""
                <div class="mc-status">
                    <span class="label">PAS</span> <span class="val">{t+1:03d}</span>  ·
                    <span class="label">ACTION</span>
                    <span class="{action_color}">{action_label}</span>  ·
                    <span class="label">RÉCOMPENSE</span>
                    <span class="val-red">{r:+.2f}</span>  ·
                    <span class="label">ARRIVÉES</span>
                    <span class="val">{info.arrivals}</span>  ·
                    <span class="label">DÉPARTS</span>
                    <span class="val">{info.departures}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            s = s_prime
            time.sleep(1.0 / speed)
            if done:
                break

        plt.close(fig)

        # Récapitulatif final
        section_title("Bilan de l'épisode")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(kpi_card("Récompense cumulée", f"{cum_reward:.1f}",
                                 color="amber"), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_card("Attente cumulée", env.cumulative_waiting,
                                 unit=" véh·pas"), unsafe_allow_html=True)
        with c3:
            avg_q = np.mean(history["queue_total"])
            st.markdown(kpi_card("File moyenne", f"{avg_q:.1f}",
                                 unit=" véh."), unsafe_allow_html=True)

        section_title("Évolution temporelle")
        df_hist = pd.DataFrame(history).set_index("step")
        st.line_chart(df_hist, color=[CYAN, AMBER])


def page_compare():
    section_title("Comparaison des politiques")
    st.markdown(
        f"<p style='color:#7E97B5;'>Évaluation sur N épisodes pour comparer "
        f"l'agent Q-Learning aux baselines. Mêmes graines aléatoires pour les "
        f"trois politiques (comparaison équitable).</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        scenario = st.selectbox("Scénario", list(SCENARIOS.keys()))
    with col2:
        episodes = st.number_input("Épisodes", 5, 200, 30, step=5)
    with col3:
        fixed_period = st.slider("Période FixedTime", 2, 30, 10)

    if st.button("LANCER L'EVALUATION"):
        config = SCENARIOS[scenario]
        rng = np.random.default_rng(123)
        seeds = rng.integers(0, 1_000_000, size=episodes)

        names = ["Q-Learning", "LongestQueue", "FixedTime"]
        all_rewards = {}
        progress = st.progress(0.0)
        for idx, n in enumerate(names):
            policy, _ = make_policy(n, scenario, fixed_period=fixed_period)
            env = TrafficEnv(config, seed=0)
            rewards = []
            for ep in range(episodes):
                env.rng = np.random.default_rng(int(seeds[ep]))
                s = env.reset()
                if hasattr(policy, "reset"):
                    policy.reset()
                G = 0.0
                done = False
                while not done:
                    a = policy.choose_action(s, greedy=True)
                    s, r, done, _ = env.step(a)
                    G += r
                rewards.append(G)
            all_rewards[n] = rewards
            progress.progress((idx + 1) / len(names))
        progress.empty()

        # KPI cards : qui gagne ?
        section_title("Résultats")
        means = {n: np.mean(all_rewards[n]) for n in names}
        winner = max(means, key=means.get)
        c1, c2, c3 = st.columns(3)
        for col, name in zip([c1, c2, c3], names):
            color = "green" if name == winner else "cyan"
            col.markdown(
                kpi_card(name, f"{means[name]:.1f}",
                         unit=" ± " + f"{np.std(all_rewards[name]):.1f}",
                         color=color),
                unsafe_allow_html=True,
            )

        # Graphique comparatif
        section_title("Visualisation")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=NAVY)
        fig.subplots_adjust(wspace=0.3)
        bar_colors = [CYAN, GREEN, AMBER]
        means_list = [means[n] for n in names]
        stds_list = [np.std(all_rewards[n]) for n in names]
        bars = axes[0].bar(names, means_list, yerr=stds_list,
                           capsize=6, color=bar_colors, alpha=0.85,
                           edgecolor=WHITE, linewidth=0.5)
        axes[0].set_ylabel("Récompense moyenne")
        axes[0].set_title("Moyenne ± écart-type", color=WHITE)
        axes[0].grid(True, alpha=0.3, axis="y")

        bp = axes[1].boxplot([all_rewards[n] for n in names],
                             tick_labels=names, patch_artist=True,
                             medianprops=dict(color=WHITE, lw=2))
        for patch, c in zip(bp["boxes"], bar_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor(c)
        for whisker in bp["whiskers"]:
            whisker.set_color(WHITE)
        for cap in bp["caps"]:
            cap.set_color(WHITE)
        for flier in bp["fliers"]:
            flier.set_markeredgecolor(CYAN)
        axes[1].set_ylabel("Récompense par épisode")
        axes[1].set_title("Distribution sur les épisodes", color=WHITE)
        axes[1].grid(True, alpha=0.3, axis="y")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def page_training():
    section_title("Courbes d'apprentissage")
    st.markdown(
        f"<p style='color:#7E97B5;'>Convergence du Q-Learning : récompense "
        f"par épisode, décroissance d'epsilon et erreur de Bellman moyenne.</p>",
        unsafe_allow_html=True,
    )

    metrics_files = [f for f in os.listdir(RESULTS_DIR)
                     if f.startswith("metrics_") and f.endswith(".json")] \
        if os.path.exists(RESULTS_DIR) else []

    if not metrics_files:
        st.warning("Aucune métrique trouvée. Lance d'abord "
                   "`python scripts/train.py --scenario <nom>`.")
        return

    scenarios = sorted(f.replace("metrics_", "").replace(".json", "")
                       for f in metrics_files)
    sel = st.selectbox("Scénario", scenarios)

    metrics = load_metrics(os.path.join(RESULTS_DIR, f"metrics_{sel}.json"))
    rewards = np.array(metrics["rewards"])
    epsilons = np.array(metrics["epsilons"])
    tds = np.array(metrics["td_errors"])

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Épisodes", metrics["episodes"]), unsafe_allow_html=True)
    c2.markdown(kpi_card("|S| visités", metrics["n_states_seen"], color="amber"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("γ (discount)", metrics['gamma']), unsafe_allow_html=True)
    c4.markdown(kpi_card("α (learning rate)", metrics['alpha']),
                unsafe_allow_html=True)

    section_title("Convergence")

    def moving_average(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, facecolor=NAVY)

    axes[0].plot(rewards, alpha=0.25, color=CYAN, lw=0.8,
                 label="Récompense / épisode")
    ma = moving_average(rewards, w=50)
    axes[0].plot(np.arange(len(ma)) + 50, ma, color=AMBER, lw=2.2,
                 label="Moyenne mobile (50 ép.)")
    axes[0].set_ylabel("Récompense")
    axes[0].set_title(f"Convergence — scénario : {sel}", color=WHITE)
    axes[0].legend(facecolor=NAVY_2, edgecolor=NAVY_3, labelcolor=WHITE)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epsilons, color=GREEN, lw=2)
    axes[1].fill_between(range(len(epsilons)), 0, epsilons,
                         color=GREEN, alpha=0.15)
    axes[1].set_ylabel(r"$\epsilon$")
    axes[1].set_title("Décroissance de l'exploration", color=WHITE)
    axes[1].grid(alpha=0.3)

    axes[2].plot(tds, alpha=0.4, color=PURPLE, lw=0.8)
    ma_td = moving_average(tds, w=50)
    axes[2].plot(np.arange(len(ma_td)) + 50, ma_td, color=PURPLE, lw=2.2,
                 label="MA(50)")
    axes[2].set_ylabel(r"$|\delta|$ TD")
    axes[2].set_xlabel("Épisode")
    axes[2].set_title("Erreur de Bellman moyenne", color=WHITE)
    axes[2].legend(facecolor=NAVY_2, edgecolor=NAVY_3, labelcolor=WHITE)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# =================================================================== Main

def main():
    st.set_page_config(
        page_title="Trafic Urbain — Mission Control",
        page_icon=":vertical_traffic_light:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Injecte le CSS Mission Control
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header principal
    render_header(
        "Gestion intelligente du trafic urbain",
        "Mission Control · Agent Q-Learning · Carrefour 4 branches",
    )

    # Sidebar
    st.sidebar.markdown(
        f"<div style='color:{CYAN}; font-weight:700; letter-spacing:2px; "
        f"font-size:0.85rem; text-transform:uppercase;'>Navigation</div>",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        " ",
        ["Simulation live", "Comparaison politiques", "Courbes d'apprentissage"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='color:#7E97B5; font-size:0.78rem; line-height:1.6;'>"
        f"<b style='color:{CYAN};'>STATUT</b><br>"
        f"<span class='mc-pulse' style='display:inline-block; vertical-align:middle;'></span> "
        f"Système opérationnel<br><br>"
        f"<b style='color:{CYAN};'>VERSION</b><br>"
        f"v1.0 — Partie 1<br><br>"
        f"<b style='color:{CYAN};'>COURS</b><br>"
        f"IAD & SMA · Pr. QAFFOU"
        f"</div>",
        unsafe_allow_html=True,
    )

    if page == "Simulation live":
        page_live()
    elif page == "Comparaison politiques":
        page_compare()
    else:
        page_training()


if __name__ == "__main__":
    main()
