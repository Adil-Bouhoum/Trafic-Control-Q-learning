"""
utils.py
--------
Fonctions utilitaires partagées par les autres modules :
- Discrétisation des files d'attente (passage continu -> espace d'états fini)
- Sauvegarde / chargement de la Q-table
- Helpers de logging et de seed pour reproductibilité
"""

import json
import os
import pickle
import random
from typing import Tuple

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Fixe les graines aléatoires (numpy + random standard) pour reproductibilité
    des expériences (arrivées Poisson, exploration epsilon-greedy, etc.).
    """
    np.random.seed(seed)
    random.seed(seed)


def discretize_queue(queue_length: int, max_level: int = 4) -> int:
    """
    Discrétise la longueur d'une file d'attente en un niveau borné [0, max_level].
    Toute valeur >= max_level est saturée à max_level.

    Pourquoi discrétiser ?
    -> Le Q-Learning tabulé exige un espace d'états FINI. Sans discrétisation,
       l'espace serait potentiellement infini (file = entier non borné).
    -> max_level=4 donne 5 niveaux par branche : {0,1,2,3,4+}.
       Avec 4 branches + 2 phases : |S| = 5^4 * 2 = 1250 états (raisonnable).
    """
    if queue_length < 0:
        raise ValueError("La longueur de file ne peut pas être négative.")
    return min(queue_length, max_level)


def encode_state(queues: Tuple[int, int, int, int],
                 phase: int,
                 max_level: int = 4) -> Tuple[int, int, int, int, int]:
    """
    Encode l'état du carrefour sous la forme d'un tuple immuable
    (utilisable comme clé de dictionnaire pour la Q-table).

    Args:
        queues : (file_N, file_S, file_E, file_O) en nombre de véhicules
        phase  : 0 = NS vert, 1 = EO vert
        max_level : niveau max de saturation pour la discrétisation

    Returns:
        Tuple (qN, qS, qE, qO, phase) discrétisé
    """
    qN, qS, qE, qO = queues
    return (
        discretize_queue(qN, max_level),
        discretize_queue(qS, max_level),
        discretize_queue(qE, max_level),
        discretize_queue(qO, max_level),
        phase,
    )


def save_q_table(q_table: dict, path: str) -> None:
    """
    Sauvegarde la Q-table au format pickle.
    On convertit explicitement en dict standard pour éviter les problèmes
    de sérialisation des defaultdict avec factory lambda.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plain = {state: np.asarray(q) for state, q in q_table.items()}
    with open(path, "wb") as f:
        pickle.dump(plain, f)


def load_q_table(path: str) -> dict:
    """Charge une Q-table sauvegardée."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: dict, path: str) -> None:
    """
    Sauvegarde les métriques d'entraînement (récompenses par épisode,
    epsilon, etc.) au format JSON pour analyse ultérieure.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Conversion explicite des np.array -> list pour JSON
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in metrics.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def load_metrics(path: str) -> dict:
    """Charge les métriques depuis un fichier JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
