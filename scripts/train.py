"""
train.py
--------
Script d'entraînement de l'agent Q-Learning.

Implémente la boucle d'apprentissage du cours (Apprentissage_Renforcement.pptx,
slide 33) :

    Initialiser Q(s,a) = 0
    Pour chaque épisode :
        s = état_initial
        Tant que s n'est pas terminal :
            Choisir a (epsilon-greedy)
            Exécuter a, observer r et s'
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
            s <- s'
        Decay de epsilon

Sauvegarde :
- Q-table       : results/q_table_<scenario>.pkl
- Métriques     : results/metrics_<scenario>.json (récompense, eps, etc. par ép.)
- Courbes       : results/convergence_<scenario>.png

Usage :
    python scripts/train.py                   # scénario "equilibre" par défaut
    python scripts/train.py --scenario asymetrique_NS --episodes 1500
"""

import argparse
import os
import sys

# Ajout du dossier parent pour pouvoir importer src/...
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (export PNG uniquement)
import matplotlib.pyplot as plt
import numpy as np

from src.agent import QLearningAgent
from src.environment import SCENARIOS, TrafficEnv
from src.utils import save_metrics, save_q_table, set_global_seed


# ------------------------------------------------------------------ Parsing

def parse_args():
    p = argparse.ArgumentParser(description="Entraînement Q-Learning")
    p.add_argument("--scenario", default="equilibre",
                   choices=list(SCENARIOS.keys()),
                   help="Scénario de trafic (cf. src/environment.py).")
    p.add_argument("--episodes", type=int, default=1000,
                   help="Nombre d'épisodes d'entraînement.")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


# ----------------------------------------------------------- Boucle d'entr.

def train(args) -> dict:
    """
    Effectue la boucle d'entraînement et retourne un dict de métriques.
    """
    set_global_seed(args.seed)

    config = SCENARIOS[args.scenario]
    env = TrafficEnv(config, seed=args.seed)
    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    rewards_per_episode = []
    epsilons = []
    td_errors_per_episode = []   # erreur TD moyenne par épisode

    print(f"=== Entraînement Q-Learning ===")
    print(f"Scénario : {args.scenario} | lambdas={config.lambdas}")
    print(f"Hyperparams : alpha={args.alpha} gamma={args.gamma} "
          f"eps0={args.epsilon} eps_min={args.epsilon_min} "
          f"decay={args.epsilon_decay}")
    print(f"Épisodes : {args.episodes}\n")

    for ep in range(1, args.episodes + 1):
        s = env.reset()
        total_reward = 0.0
        total_td = 0.0
        n_steps = 0
        done = False

        while not done:
            a = agent.choose_action(s)
            s_prime, r, done, _ = env.step(a)
            td = agent.update(s, a, r, s_prime, done)
            s = s_prime
            total_reward += r
            total_td += abs(td)
            n_steps += 1

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        epsilons.append(agent.epsilon)
        td_errors_per_episode.append(total_td / max(n_steps, 1))

        if ep % max(1, args.episodes // 20) == 0 or ep == 1:
            # Moyenne glissante sur les 50 derniers épisodes pour lisser le bruit
            recent = np.mean(rewards_per_episode[-50:])
            print(f"  Ep {ep:5d}/{args.episodes}  "
                  f"G={total_reward:8.2f}  "
                  f"G_avg(50)={recent:8.2f}  "
                  f"eps={agent.epsilon:.3f}  "
                  f"|TD|_avg={td_errors_per_episode[-1]:.3f}  "
                  f"|S|={agent.n_states_seen()}")

    metrics = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon0": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "rewards": rewards_per_episode,
        "epsilons": epsilons,
        "td_errors": td_errors_per_episode,
        "n_states_seen": agent.n_states_seen(),
    }

    # Sauvegardes
    os.makedirs(args.results_dir, exist_ok=True)
    save_q_table(agent.Q, os.path.join(args.results_dir, f"q_table_{args.scenario}.pkl"))
    save_metrics(metrics, os.path.join(args.results_dir, f"metrics_{args.scenario}.json"))
    plot_convergence(metrics, os.path.join(args.results_dir, f"convergence_{args.scenario}.png"))

    print(f"\n[OK] Q-table -> {args.results_dir}/q_table_{args.scenario}.pkl")
    print(f"[OK] Métriques -> {args.results_dir}/metrics_{args.scenario}.json")
    print(f"[OK] Courbe -> {args.results_dir}/convergence_{args.scenario}.png")
    return metrics


# ---------------------------------------------------------- Visualisation

def plot_convergence(metrics: dict, out_path: str) -> None:
    """
    Génère la figure de convergence avec 3 sous-graphes :
      1. Récompense par épisode + moyenne mobile (50 ép.)
      2. Évolution de epsilon
      3. Erreur TD moyenne par épisode (proxy de convergence)
    """
    rewards = np.array(metrics["rewards"])
    epsilons = np.array(metrics["epsilons"])
    tds = np.array(metrics["td_errors"])

    # Moyenne mobile pour lisser
    def moving_average(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(rewards, alpha=0.3, label="Récompense / épisode")
    ma = moving_average(rewards, w=50)
    axes[0].plot(np.arange(len(ma)) + 50, ma, color="C1", linewidth=2,
                 label="Moyenne mobile (50 ép.)")
    axes[0].set_ylabel("Récompense cumulée")
    axes[0].set_title(f"Convergence Q-Learning — scénario : {metrics['scenario']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epsilons, color="C2")
    axes[1].set_ylabel(r"$\epsilon$")
    axes[1].set_title("Décroissance de l'exploration")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(tds, color="C3", alpha=0.6)
    ma_td = moving_average(tds, w=50)
    axes[2].plot(np.arange(len(ma_td)) + 50, ma_td, color="darkred", linewidth=2,
                 label="Moyenne mobile (50 ép.)")
    axes[2].set_ylabel(r"$|\delta|$ TD moyenne")
    axes[2].set_xlabel("Épisode")
    axes[2].set_title("Erreur de Bellman moyenne par épisode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------- Main

if __name__ == "__main__":
    train(parse_args())
