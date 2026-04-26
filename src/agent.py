"""
agent.py
--------
Agent Q-Learning tabulé, implémenté FROM SCRATCH (contrainte du sujet §4.1).

Référence du cours (Pr. QAFFOU - Apprentissage_Renforcement.pptx) :
- Slide 23 : Équation de Bellman optimale pour Q*(s,a)
- Slide 31 : Équation de mise à jour Q-Learning :
                Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
- Slide 32 : Politique epsilon-greedy (exploration / exploitation)
- Slide 33 : Pseudo-code de référence

Choix d'implémentation :
- Q-table sous forme de defaultdict(np.zeros) : pas besoin de pré-allouer tous
  les états, les inconnus sont initialisés à 0 à la volée. Économise la mémoire
  et reste cohérent avec l'initialisation du cours ("Q(s,a) = 0 pour tous s, a").
- Decay de epsilon : exponentiel (geometric decay) — exploration forte au début,
  exploitation pure à la fin.
"""

from typing import Dict, Tuple

import numpy as np

from src.environment import N_ACTIONS


class QLearningAgent:
    """
    Agent Q-Learning tabulé (off-policy, model-free).

    Hyperparamètres (justifiés dans le rapport) :
      alpha   : taux d'apprentissage (0.1 par défaut)
                - trop grand -> oscillations
                - trop petit -> convergence lente
      gamma   : facteur d'actualisation (0.95)
                - on valorise les récompenses futures (horizon long du trafic)
                - 0.95 : bon compromis entre myopie et infinitude
      epsilon : taux d'exploration initial (1.0)
      epsilon_min     : plancher d'exploration (0.05) - on garde toujours un peu
                        d'aléa pour ne pas se figer dans un optimum local
      epsilon_decay   : facteur de décroissance par épisode (0.995)
                        Après ~600 épisodes : epsilon ~ 0.05
    """

    def __init__(self,
                 n_actions: int = N_ACTIONS,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 seed: int = None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table : dict[état -> np.array de taille n_actions]
        # Initialisation à 0 à la volée via _get_q (cf. slide 33 du cours).
        # On utilise un dict standard (pas defaultdict) pour rester pickleable
        # sans recourir à une factory module-level.
        self.Q: Dict[Tuple, np.ndarray] = {}

        # Générateur aléatoire local (reproductibilité de l'exploration)
        self.rng = np.random.default_rng(seed)

        # Historique pour analyse de convergence
        self.last_td_error: float = 0.0

    # -------------------------------------------------- Helper Q-table

    def _get_q(self, state: Tuple) -> np.ndarray:
        """
        Renvoie le vecteur Q[s] pour l'état s. Si l'état est inconnu, l'initialise
        à zéro (équivalent au "Q(s,a) = 0 pour tous s, a" du cours, slide 33).
        """
        q = self.Q.get(state)
        if q is None:
            q = np.zeros(self.n_actions, dtype=np.float64)
            self.Q[state] = q
        return q

    # -------------------------------------------------- Choix d'action

    def choose_action(self, state: Tuple, greedy: bool = False) -> int:
        """
        Sélectionne une action selon la politique epsilon-greedy (slide 32).

        Args:
            state  : état courant (tuple)
            greedy : si True, force l'exploitation pure (utilisé à l'évaluation,
                     pas pendant l'entraînement).

        Returns:
            action choisie (int dans [0, n_actions[)
        """
        if not greedy and self.rng.random() < self.epsilon:
            # Exploration : action aléatoire uniforme
            return int(self.rng.integers(0, self.n_actions))
        # Exploitation : argmax_a Q(s, a)
        # En cas d'égalité, on tire au hasard parmi les meilleures actions
        # pour éviter un biais systématique sur la première action.
        q_values = self._get_q(state)
        max_q = q_values.max()
        best = np.flatnonzero(q_values == max_q)
        return int(self.rng.choice(best))

    # -------------------------------------------------- Mise à jour Q

    def update(self,
               state: Tuple,
               action: int,
               reward: float,
               next_state: Tuple,
               done: bool) -> float:
        """
        Mise à jour de la Q-table selon la règle de Bellman du Q-Learning
        (slide 31 du cours) :

            Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

        Si l'épisode est terminé (done=True), il n'y a pas d'état futur :
        on prend max_a' Q(s',a') = 0.

        Args:
            state      : s
            action     : a
            reward     : r reçu après avoir exécuté a en s
            next_state : s' (état suivant observé)
            done       : True si s' est terminal

        Returns:
            erreur TD (Temporal Difference) — utile pour suivre la convergence.
        """
        q_state = self._get_q(state)
        current_q = q_state[action]

        # Cible TD : r + gamma * max_a' Q(s', a')   (0 si terminal)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self._get_q(next_state).max()

        # Erreur TD (delta de Bellman)
        td_error = td_target - current_q

        # Mise à jour incrémentale (slide 30 : "nouvelle = ancienne + alpha * erreur")
        q_state[action] = current_q + self.alpha * td_error

        self.last_td_error = float(td_error)
        return self.last_td_error

    # -------------------------------------------------- Decay d'exploration

    def decay_epsilon(self) -> None:
        """
        Décroissance géométrique de epsilon, appelée à la fin de chaque épisode.
        On ne descend jamais sous epsilon_min pour garder une exploration résiduelle.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # -------------------------------------------------- Introspection

    def get_policy(self) -> Dict[Tuple, int]:
        """
        Renvoie la politique gloutonne extraite de la Q-table :
            pi(s) = argmax_a Q(s, a)
        Utile pour la phase d'évaluation et la visualisation.
        """
        return {state: int(np.argmax(q)) for state, q in self.Q.items()}

    def n_states_seen(self) -> int:
        """Nombre d'états distincts rencontrés depuis le début de l'entraînement."""
        return len(self.Q)
