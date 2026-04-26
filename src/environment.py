"""
environment.py
--------------
Simulation du carrefour à 4 branches (Nord, Sud, Est, Ouest).

Formalisation MDP :
- État S       : (file_N, file_S, file_E, file_O, phase) discrétisé
- Action A     : 0 = Maintenir la phase, 1 = Changer la phase
- Récompense R : -(somme des files d'attente) -> incite à minimiser l'attente
                 + pénalité de changement (anti-flapping)
- Transition T : stochastique
                 - arrivées : processus de Poisson de paramètre lambda_i par branche
                 - départs  : nb fixe de véhicules écoulés sur la phase verte
- Episodique   : oui (longueur fixe = max_steps)

Cette implémentation est volontairement séparée de l'agent pour respecter la
contrainte du sujet : "module de simulation séparable de l'agent".
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.utils import encode_state


# --- Constantes de phase ---------------------------------------------------
PHASE_NS = 0   # Nord-Sud vert (Est-Ouest rouge)
PHASE_EO = 1   # Est-Ouest vert (Nord-Sud rouge)

# --- Constantes d'action ---------------------------------------------------
ACTION_KEEP = 0
ACTION_SWITCH = 1
N_ACTIONS = 2


@dataclass
class TrafficConfig:
    """
    Configuration de la simulation du carrefour.

    Attributs :
        lambdas : taux d'arrivées Poisson par branche (N, S, E, O), véhicules/pas.
                  Ex. (0.3, 0.3, 0.3, 0.3) = trafic équilibré
                      (0.5, 0.5, 0.1, 0.1) = trafic asymétrique (axe NS chargé)
        cars_per_green : nb de véhicules écoulés par pas sur une branche au vert.
        max_queue : capacité max d'une file (au-delà : véhicules perdus).
        max_steps : durée d'un épisode (en pas de temps).
        switch_penalty : pénalité ajoutée à la récompense lors d'un changement
                         de phase (modélise le coût de la transition orange).
        max_level : niveau de saturation pour la discrétisation des files.
    """
    lambdas: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)
    cars_per_green: int = 2
    max_queue: int = 20
    max_steps: int = 200
    switch_penalty: float = 0.5
    max_level: int = 4


@dataclass
class StepInfo:
    """Informations détaillées d'un pas (utiles pour le rendu Streamlit)."""
    queues: Tuple[int, int, int, int]
    phase: int
    arrivals: Tuple[int, int, int, int]
    departures: Tuple[int, int, int, int]
    reward: float
    step: int


class TrafficEnv:
    """
    Environnement de simulation du carrefour.

    API inspirée de l'API Gym (gymnasium) :
      env.reset() -> état_initial
      env.step(action) -> (état_suivant, récompense, terminé, info)

    Cette API rend l'environnement directement utilisable par un agent
    Q-Learning standard sans adaptation.
    """

    def __init__(self, config: TrafficConfig = None, seed: int = None):
        self.config = config if config is not None else TrafficConfig()
        # Générateur aléatoire local : permet la reproductibilité par-environnement
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ API

    def reset(self) -> Tuple[int, int, int, int, int]:
        """
        Réinitialise l'environnement au début d'un épisode.
        Files vides, phase initiale = NS, compteur de pas à 0.

        Returns:
            État initial discrétisé (qN, qS, qE, qO, phase).
        """
        self._queues = np.zeros(4, dtype=int)   # [N, S, E, O]
        self._phase = PHASE_NS
        self._t = 0
        self._cum_waiting = 0   # somme cumulée des files (pour métriques)
        return self._get_state()

    def step(self, action: int) -> Tuple[Tuple, float, bool, StepInfo]:
        """
        Effectue un pas de simulation.

        Ordre des opérations (important pour la cohérence du MDP) :
          1) Application de l'action (changement de phase éventuel).
          2) Arrivées stochastiques (Poisson) sur les 4 branches.
          3) Départs déterministes sur la phase verte.
          4) Calcul de la récompense.
          5) Avancement du temps + test de fin d'épisode.

        Args:
            action : 0 (Maintenir) ou 1 (Changer).

        Returns:
            (next_state, reward, done, info)
        """
        if action not in (ACTION_KEEP, ACTION_SWITCH):
            raise ValueError(f"Action invalide : {action}")

        # 1) Action : changement de phase éventuel
        switched = False
        if action == ACTION_SWITCH:
            self._phase = 1 - self._phase    # bascule NS <-> EO
            switched = True

        # 2) Arrivées Poisson par branche (vehiculea/pas, indépendantes)
        arrivals = self.rng.poisson(self.config.lambdas).astype(int)
        self._queues = np.minimum(self._queues + arrivals, self.config.max_queue)

        # 3) Départs : seules les branches au vert écoulent du trafic
        departures = np.zeros(4, dtype=int)
        if self._phase == PHASE_NS:
            departures[0] = min(self._queues[0], self.config.cars_per_green)  # N
            departures[1] = min(self._queues[1], self.config.cars_per_green)  # S
        else:  # PHASE_EO
            departures[2] = min(self._queues[2], self.config.cars_per_green)  # E
            departures[3] = min(self._queues[3], self.config.cars_per_green)  # O
        self._queues -= departures

        # 4) Récompense :
        #    - on pénalise toute file d'attente (incite à fluidifier le trafic).
        #    - on ajoute une pénalité au changement de phase pour éviter le
        #      "flapping" (basculer en boucle entre les phases).
        reward = -float(self._queues.sum())
        if switched:
            reward -= self.config.switch_penalty

        # Métriques cumulées pour analyse a posteriori
        self._cum_waiting += int(self._queues.sum())

        # 5) Avancement du temps
        self._t += 1
        done = self._t >= self.config.max_steps

        info = StepInfo(
            queues=tuple(int(x) for x in self._queues),
            phase=self._phase,
            arrivals=tuple(int(x) for x in arrivals),
            departures=tuple(int(x) for x in departures),
            reward=reward,
            step=self._t,
        )
        return self._get_state(), reward, done, info

    # ----------------------------------------------------------- Helpers

    def _get_state(self) -> Tuple[int, int, int, int, int]:
        """Renvoie l'état courant discrétisé (utilisable comme clé Q-table)."""
        return encode_state(
            tuple(int(x) for x in self._queues),
            self._phase,
            max_level=self.config.max_level,
        )

    @property
    def cumulative_waiting(self) -> int:
        """Somme cumulée des files depuis le dernier reset (métrique d'évaluation)."""
        return self._cum_waiting

    @property
    def raw_queues(self) -> np.ndarray:
        """Files réelles (non discrétisées) — utile pour le rendu live."""
        return self._queues.copy()

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def t(self) -> int:
        return self._t


# --- Scénarios de trafic prédéfinis ----------------------------------------
# Au moins deux scénarios sont demandés par le sujet (cf. §3.3 du PDF).

SCENARIOS: Dict[str, TrafficConfig] = {
    "equilibre": TrafficConfig(
        lambdas=(0.3, 0.3, 0.3, 0.3),
        cars_per_green=2,
        max_steps=200,
    ),
    "asymetrique_NS": TrafficConfig(
        lambdas=(0.6, 0.6, 0.1, 0.1),
        cars_per_green=2,
        max_steps=200,
    ),
    "asymetrique_EO": TrafficConfig(
        lambdas=(0.1, 0.1, 0.6, 0.6),
        cars_per_green=2,
        max_steps=200,
    ),
    "trafic_dense": TrafficConfig(
        lambdas=(0.5, 0.5, 0.5, 0.5),
        cars_per_green=2,
        max_steps=200,
    ),
}
