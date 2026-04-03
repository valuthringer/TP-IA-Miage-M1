# TD 1 - Puissance 4

## Question 1 : Heuristiques
### 1.1 - Explications sur les heuristiques

Dans la Question 1, les agents IA "non ml" les plus simples sont implémentés dans Question-1/p4_tools.py.
Leur principe est de choisir le coup suivant avec des règles fixes (heuristiques), sans apprentissage.

1. **Paul**
   - Heuristique : **aléatoire pur**.
   - Principe : choisit un coup au hasard parmi les colonnes valides.
   - Comportement : imprévisible, sans stratégie de victoire ni de défense.

2. **Pierre**
   - Heuristique : **priorité au centre**.
   - Principe : joue la colonne 3 (centre) si elle est disponible, sinon joue un coup aléatoire.
   - Intuition : le centre donne plus de possibilités d'alignements horizontaux, verticaux et diagonaux.

3. **JeanClaude**
   - Heuristique : **victoire immédiate d'abord**, puis centre.
   - Principe : simule chaque coup possible ; si un coup gagne tout de suite, il le joue.
   - Sinon : joue le centre si possible, sinon aléatoire.
   - Idée : opportunisme à 1 coup (pas d'anticipation profonde).

4. **Nicolas**
   - Heuristique : **blocage de la victoire adverse**, puis centre.
   - Principe : teste les coups de l'adversaire au prochain tour ; si un coup adverse est gagnant, il joue cette colonne pour bloquer.
   - Sinon : joue le centre si possible, sinon aléatoire.
   - Idée : défense à court terme (1 coup d'anticipation adverse).

5. **Sylvie**
   - Heuristique : **évaluation locale du coup** (mini-anticipation type minimax très simplifié).
    - Principe :
       - joue un coup gagnant immédiat si possible ;
       - pénalise fortement les coups qui laissent une victoire immédiate à l'adversaire ;
       - favorise les colonnes centrales (3 > 2/4 > 1/5).
    - Comportement : plus "stratégique" que les autres heuristiques simples, tout en restant non-ML.

En résumé, ces agents vont d'une logique très basique (hasard) à une logique à courte profondeur (gagner tout de suite, bloquer, privilégier le centre), mais **aucun n'apprend à partir de données** : toutes les règles sont codées à la main.

### 1.2 - Classement des agents

Rang | Agent       | Pts | J   | G  | N | P
-----+-------------+-----+-----+----+---+---
   1 | Emmanuelle  | 300 | 100 | 100| 0 | 0
   2 | Sylvie      | 205 | 100 | 68 | 1 | 31
   3 | Sylvain     | 198 | 100 | 65 | 3 | 32
   4 | Isabelle    | 191 | 100 | 62 | 5 | 33
   5 | Nicolas     | 160 | 100 | 51 | 7 | 42
   6 | JeanClaude  | 153 | 100 | 51 | 0 | 49
   7 | Gilles      | 111 | 100 | 37 | 0 | 63
   8 | Nathalie    |  98 | 100 | 32 | 2 | 66
   9 | Pierre      |  55 | 100 | 18 | 1 | 81
  10 | Paul        |  19 | 100 | 6  | 1 | 93

### 2 - Agent 6 (custom)

L'apprentissage a été fait en deux phases :

1. **Curriculum contre agents non-ML**
    - Le modèle est entraîné successivement contre plusieurs adversaires heuristiques (Paul, Pierre, JeanClaude, Nicolas, Sylvie, Isabelle, Gilles, Nathalie, Emmanuelle, Sylvain).
    - L'idée est de commencer par des adversaires simples puis d'augmenter la difficulté.
    - Cela permet d'apprendre progressivement : coups valides, menaces directes, blocages, contrôle du centre, etc.

2. **Self-play par snapshots figés**
    - Ensuite, le modèle joue contre des versions sauvegardées de lui-même (snapshots).
    - À chaque cycle : sauvegarde d'un snapshot, entraînement contre ce snapshot, puis nouvelle évaluation.
    - Cette phase améliore la robustesse et limite la dépendance à une seule stratégie adverse.

#### Choix techniques

- Algorithme : **DQN** (policy `MlpPolicy`).
- Environnement : `Puissance4GymnasiumEnv` (`p4_env_gymnasium.py`).
- Randomisation utile : `adversaire_commence_aleatoire=True` pour réduire le biais "premier joueur".
- Paramètres principaux (script custom) :
   - `learning_rate = 2.5e-4`
   - `buffer_size = 300000`
   - `batch_size = 256`
   - `gamma = 0.99`
   - `exploration_initial_eps = 1.0`
   - `exploration_final_eps = 0.03`
   - `exploration_fraction = 0.35`

#### Évaluation pendant l'entraînement

Après chaque bloc d'entraînement, le modèle est évalué contre **tous** les agents non-ML (plusieurs parties par adversaire).

- Score global utilisé :
   - Victoire = 3 points
   - Nul = 1 point
   - Défaite = 0 point
- Un modèle "best" est sauvegardé dès que ce score global s'améliore.

Ainsi, on évite de garder un modèle qui performe seulement sur un adversaire particulier.

Fichiers de sortie

- Modèle courant : `sb3_sorties/dqn_puissance4_champion.zip`
- Meilleur modèle global : `sb3_sorties/dqn_puissance4_champion_best.zip`
- Snapshots de self-play : dossier `sb3_sorties/snapshots_champion/`

#### Commandes utilisées

#### Entraîner l'agent custom :

```bash
python main_6_agent_champion.py
```

#### Lancer le championnat avec le meilleur modèle :

```bash
python championnat.py sb3_sorties/dqn_puissance4_champion_best.zip
```