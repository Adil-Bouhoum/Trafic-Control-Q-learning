"""
baseline.py
-----------
Politiques de référence (baselines) pour comparer la performance du Q-Learning.

Le sujet (§3.2 question 6) demande au minimum UNE politique baseline. On en
fournit deux pour rendre la comparaison plus rigoureuse :

1. FixedTimePolicy : feu à durée fixe (alternance périodique).
   -> stratégie classique des feux non-adaptatifs (la plupart des carrefours
      réels sans capteurs). C'est notre baseline principale.

2. LongestQueuePolicy : règle heuristique gloutonne (donne le vert à l'axe
   ayant la plus longue file). Sert de borne supérieure "raisonnable" non
   apprise — utile pour situer la performance du Q-Learning.

Les deux politiques exposent la même API que l'agent Q-Learning :
    policy.choose_action(state) -> action
ce qui les rend interchangeables dans les scripts d'évaluation.
"""

from typing import Tuple

from src.environment import ACTION_KEEP, ACTION_SWITCH, PHASE_NS, PHASE_EO


class FixedTimePolicy:
    """
    Politique à durée fixe : la phase change tous les `period` pas de temps,
    indépendamment de l'état des files.

    C'est le mode de fonctionnement par défaut des feux non-intelligents.
    Justification : c'est la baseline la plus représentative du "monde réel"
    avant l'introduction d'un contrôle adaptatif.

    Args:
        period : nb de pas avant chaque changement de phase (ex. 10).
    """

    def __init__(self, period: int = 10):
        if period < 1:
            raise ValueError("La période doit être >= 1.")
        self.period = period
        self._counter = 0

    def reset(self) -> None:
        """À appeler en début d'épisode."""
        self._counter = 0

    def choose_action(self, state: Tuple, greedy: bool = True) -> int:
        """
        Renvoie SWITCH tous les `period` pas, sinon KEEP.
        L'argument `state` est ignoré (politique non adaptative).
        L'argument `greedy` est ignoré (politique déterministe).
        """
        self._counter += 1
        if self._counter >= self.period:
            self._counter = 0
            return ACTION_SWITCH
        return ACTION_KEEP


class LongestQueuePolicy:
    """
    Politique heuristique gloutonne : donne le vert à l'axe (NS ou EO)
    ayant la plus longue somme de files d'attente.

    Plus "intelligente" qu'une durée fixe sans pour autant apprendre.
    Sert de point de comparaison pour montrer ce qu'un Q-Learning bien
    entraîné apporte par rapport à une simple règle réactive.

    Note pédagogique : on peut voir cette politique comme une architecture
    REACTIVE PURE (cf. cours sur les architectures d'agents). Le Q-Learning
    correspond plutôt à une architecture HYBRIDE (perception + apprentissage
    + délibération via Q-table).
    """

    def reset(self) -> None:
        """No-op : la politique est sans mémoire (purement réactive)."""
        pass

    def choose_action(self, state: Tuple, greedy: bool = True) -> int:
        """
        State = (qN, qS, qE, qO, phase) discrétisé.

        Compare la charge NS (qN+qS) et EO (qE+qO) :
        - Si l'axe vert a déjà la plus grosse charge -> KEEP
        - Sinon -> SWITCH
        En cas d'égalité : KEEP (évite le flapping inutile).
        """
        qN, qS, qE, qO, phase = state
        load_NS = qN + qS
        load_EO = qE + qO

        if phase == PHASE_NS:
            return ACTION_SWITCH if load_EO > load_NS else ACTION_KEEP
        else:  # PHASE_EO
            return ACTION_SWITCH if load_NS > load_EO else ACTION_KEEP
