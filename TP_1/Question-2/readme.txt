Projet - exemples progressifs d'entrainement SB3 (Puissance 4)

Ce dossier contient 5 scripts `main_*.py` pour montrer une progression simple de l'usage de l'agent RL.

1) main_1.py
- Crée un nouveau modèle DQN.
- Entraine ce modèle contre un agent non-ML (Paul).
- Teste le modèle.
- Sauvegarde le modèle.

2) main_2.py
- Charge un modèle déjà sauvegardé.
- Continue l'entrainement contre un autre agent non-ML (Pierre).
- Teste avant/après.
- Resauvegarde le modèle.

3) main_3.py
- Charge un modèle sauvegardé comme adversaire "snapshot" (figé).
- Crée un nouveau modèle à entrainer.
- Entraine le nouveau modèle contre ce snapshot.
- Teste et sauvegarde le nouveau modèle.

4) main_4.py
- Charge un modèle existant.
- Fait une évaluation avant entrainement.
- Lance une phase de self-play (le modèle joue contre lui-même).
- Fait une évaluation après entrainement.

5) main_5.py
- Met en place une boucle d'entrainement par snapshots successifs.
- A chaque cycle: entrainement -> test -> sauvegarde snapshot -> rechargement du snapshot comme adversaire.
- Permet d'illustrer une version simple de curriculum en self-play.

6) championnat.py
- Charge un modèle sauvegardé (argument en ligne de commande, avec extension `.zip`).
- Crée un championnat "tous contre tous" entre ce modèle et les agents non-ML.
- Joue des duels pour toutes les paires d'agents.
- Affiche un classement final trié par nombre de victoires décroissant.
- Exemple: `python championnat.py sb3_sorties/dqn_puissance4_v2.zip`

Notes:
- Les adversaires sont créés via `AdversaireFactory` (fichier `p4_adversaires.py`).
- L'environnement Gymnasium est construit via `creer_env` (fichier `p4_env_gymnasium.py`).
- Les modèles/snapshots sont enregistrés dans `sb3_sorties/`.
