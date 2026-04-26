"""
evaluate.py
-----------
Script d'évaluation : compare l'agent Q-Learning entraîné aux baselines
(FixedTime + LongestQueue) sur plusieurs scénarios de trafic.

Sortie :
- Tableau récapitulatif (récompense moyenne, attente cumulée moyenne)
- Graphique comparatif : results/comparison_<scenario>.png

Le sujet (§3.3 q.8) impose au moins 2 scénarios. Par défaut on évalue sur les
4 scénarios définis dans environment.py.

Usage :
    python scripts/evaluate.py
    python scripts/evaluate.py --scenario asymetrique_NS --episodes 30
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agent import QLearningAgent
from src.baseline import FixedTimePolicy, LongestQueuePolicy
from src.environment import SCENARIOS, TrafficEnv
from src.utils import load_q_table, set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Évaluation Q-Learning vs baselines")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()),
                   help="Scénarios à évaluer.")
    p.add_argument("--episodes", type=int, default=30,
                   help="Nb d'épisodes d'évaluation par scénario.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--fixed-period", type=int, default=10)
    return p.parse_args()


def run_policy(env: TrafficEnv, policy, episodes: int, seeds) -> dict:
    """
    Exécute `policy` (objet ayant .choose_action et .reset) pour `episodes`
    épisodes. Retourne récompense + attente cumulée par épisode.
    """
    rewards, waits = [], []
    for ep in range(episodes):
        # Re-seed l'env pour chaque épisode -> mêmes arrivées pour chaque
        # politique (comparaison équitable).
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
        waits.append(env.cumulative_waiting)
    return {"rewards": rewards, "waiting": waits}


class _QPolicyWrapper:
    """Adapte un QLearningAgent à l'API .choose_action(state, greedy=True)."""

    def __init__(self, q_table: dict):
        self.agent = QLearningAgent()
        # On remplace la Q-table par celle chargée
        for k, v in q_table.items():
            self.agent.Q[k] = v

    def reset(self):
        pass

    def choose_action(self, state, greedy=True):
        return self.agent.choose_action(state, greedy=True)


def evaluate_scenario(scenario: str, episodes: int, seed: int,
                      results_dir: str, fixed_period: int) -> pd.DataFrame:
    """Évalue les 3 politiques sur un scénario donné."""
    set_global_seed(seed)
    config = SCENARIOS[scenario]

    # Charge la Q-table entraînée pour ce scénario (sinon, on retombe sur
    # une Q-table vide -> agent quasi aléatoire en évaluation : utile pour
    # vérifier que la baseline est meilleure qu'un agent non entraîné).
    q_path = os.path.join(results_dir, f"q_table_{scenario}.pkl")
    if os.path.exists(q_path):
        q_table = load_q_table(q_path)
        print(f"[{scenario}] Q-table chargée ({len(q_table)} états).")
    else:
        q_table = {}
        print(f"[{scenario}] AUCUNE Q-table trouvée -> agent non entraîné !")

    # Mêmes seeds par épisode pour les 3 politiques (comparaison équitable)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000, size=episodes)

    policies = {
        "Q-Learning": _QPolicyWrapper(q_table),
        "FixedTime": FixedTimePolicy(period=fixed_period),
        "LongestQueue": LongestQueuePolicy(),
    }

    results = {}
    for name, pol in policies.items():
        env = TrafficEnv(config, seed=seed)
        results[name] = run_policy(env, pol, episodes, seeds)

    # Tableau récap
    rows = []
    for name, res in results.items():
        rows.append({
            "scenario": scenario,
            "policy": name,
            "G_mean": np.mean(res["rewards"]),
            "G_std": np.std(res["rewards"]),
            "wait_mean": np.mean(res["waiting"]),
            "wait_std": np.std(res["waiting"]),
        })
    df = pd.DataFrame(rows)

    # Graphique comparatif
    plot_comparison(scenario, results, results_dir)

    return df


def plot_comparison(scenario: str, results: dict, results_dir: str) -> None:
    """Bar plot + box plot de comparaison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    names = list(results.keys())
    means = [np.mean(results[n]["rewards"]) for n in names]
    stds = [np.std(results[n]["rewards"]) for n in names]

    axes[0].bar(names, means, yerr=stds, capsize=6,
                color=["C0", "C1", "C2"], alpha=0.85)
    axes[0].set_ylabel("Récompense moyenne (± écart-type)")
    axes[0].set_title(f"Comparaison des politiques — {scenario}")
    axes[0].grid(True, alpha=0.3, axis="y")

    data = [results[n]["rewards"] for n in names]
    axes[1].boxplot(data, labels=names)
    axes[1].set_ylabel("Récompense par épisode")
    axes[1].set_title("Distribution sur les épisodes d'évaluation")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"comparison_{scenario}.png"), dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    all_dfs = []
    for sc in args.scenarios:
        if sc not in SCENARIOS:
            print(f"[WARN] scénario inconnu : {sc} (ignoré)")
            continue
        df = evaluate_scenario(sc, args.episodes, args.seed,
                               args.results_dir, args.fixed_period)
        all_dfs.append(df)
        print(df.to_string(index=False))
        print()

    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        out_csv = os.path.join(args.results_dir, "evaluation_summary.csv")
        full.to_csv(out_csv, index=False)
        print(f"\n[OK] Résumé global -> {out_csv}")
        print("\n=== Récapitulatif ===")
        print(full.to_string(index=False))


if __name__ == "__main__":
    main()
