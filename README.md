# Gestion intelligente du trafic urbain — Partie 1

Projet IAD & SMA — Année 2025-2026
Cours : Apprentissage par Renforcement (Pr. QAFFOU) — Cycle Ingénieur

Agent Q-Learning tabulé contrôlant un feu de signalisation à un carrefour
isolé à 4 branches (Nord, Sud, Est, Ouest), implémenté **from scratch**
(contrainte du sujet §4.1).

## Structure du projet

```
.
├── src/                          # Modules métier
│   ├── environment.py            # Simulation du carrefour (MDP, Poisson, files)
│   ├── agent.py                  # Agent Q-Learning (from scratch)
│   ├── baseline.py               # FixedTime + LongestQueue (références)
│   └── utils.py                  # Discrétisation, save/load
├── scripts/
│   ├── train.py                  # Boucle d'apprentissage + courbes
│   └── evaluate.py               # Comparaison Q-Learning vs baselines
├── app/
│   └── streamlit_app.py          # Démonstration live + visualisation
├── results/                      # Q-tables, métriques JSON, courbes PNG
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate            # Linux/Mac
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner l'agent

```bash
# Scénario par défaut (trafic équilibré)
python scripts/train.py

# Trafic asymétrique avec plus d'épisodes
python scripts/train.py --scenario asymetrique_NS --episodes 1500
```

Scénarios disponibles : `equilibre`, `asymetrique_NS`, `asymetrique_EO`, `trafic_dense`.

Sortie dans `results/` :
- `q_table_<scenario>.pkl`        (Q-table sérialisée)
- `metrics_<scenario>.json`       (récompense, ε, erreur TD par épisode)
- `convergence_<scenario>.png`    (courbe de convergence)

### 2. Évaluer / comparer

```bash
python scripts/evaluate.py --scenarios equilibre asymetrique_NS --episodes 30
```

Génère un tableau récapitulatif et `results/comparison_<scenario>.png`.

### 3. Démo live (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Trois pages :
- **Simulation live** : carrefour dessiné en temps réel, voitures qui s'accumulent et s'écoulent
- **Comparaison** : barplot des politiques sur N épisodes
- **Courbes** : convergence du Q-Learning

## Modélisation MDP

| Composant | Choix |
|---|---|
| États S | (qN, qS, qE, qO, phase) discrétisés à 5 niveaux/branche → \|S\| = 5⁴×2 = 1250 |
| Actions A | {Maintenir, Changer phase} |
| Récompense R | -Σ(files d'attente) - 0.5 × switch (anti-flapping) |
| γ (actualisation) | 0.95 |
| Dynamique T | Stochastique (arrivées Poisson λ_i), inconnue → justifie Q-Learning |
| Architecture | Hybride : perception (état) + apprentissage (Q-table) + exploitation |

## Hyperparamètres Q-Learning

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| α (learning rate) | 0.1 | Compromis stabilité / vitesse |
| γ (discount) | 0.95 | Horizon long, on valorise le débit futur |
| ε₀ | 1.0 | Exploration totale au démarrage |
| ε_min | 0.05 | Exploration résiduelle (évite l'enfermement) |
| ε_decay | 0.995 | Atteint ε_min après ~600 épisodes |

## Résultats (extrait, 800 épisodes par scénario)

| Scénario | Q-Learning | LongestQueue | FixedTime |
|---|---|---|---|
| Équilibré (λ=0.3 partout) | -217 ± 53 | -183 ± 16 | -790 ± 83 |
| Asymétrique NS (λ_NS=0.6, λ_EO=0.1) | **-127 ± 21** | -139 ± 21 | -1070 ± 126 |

L'agent dépasse même la baseline LongestQueue sur le trafic asymétrique, en
exploitant la régularité statistique du flux dominant.
