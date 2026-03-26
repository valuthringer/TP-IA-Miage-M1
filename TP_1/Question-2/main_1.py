# entrainer_sb3.py
# Entraînement DQN SB3 sur Puissance4GymnasiumEnv (adversaire par défaut: Paul).
#
# Usage :
#   python entrainer_sb3.py
#
# Dépendances :
#   pip install gymnasium stable-baselines3 torch
#
# Notes TP :
# - L'env fait un "tour complet" : coup de l'agent puis coup de l'adversaire.
# - Reward differee : +1 / -1 / 0 en fin de partie, sinon 0 (sauf pénalité actions invalides).

from __future__ import annotations

import os
from typing import Tuple

from p4_adversaires import AdversaireFactory
from p4_env_gymnasium import Puissance4GymnasiumEnv, creer_env
from p4_tools import Paul

from stable_baselines3 import DQN

def evaluer_modele(model, env: Puissance4GymnasiumEnv, nb_parties: int) -> Tuple[float, float, float]:
    """Retourne (taux_victoire, taux_defaite, taux_nul) vs l'adversaire de l'env."""
    vic, defa, nul = 0, 0, 0

    # On force la politique déterministe pendant l'évaluation
    for _ in range(nb_parties):
        obs, info = env.reset()
        termine = False
        while not termine:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            termine = bool(terminated or truncated)

        vainqueur = info.get("vainqueur", None)
        if vainqueur == +1:
            vic += 1
        elif vainqueur == -1:
            defa += 1
        else:
            nul += 1

    n = max(1, nb_parties)
    return vic / n, defa / n, nul / n


def main():
    os.makedirs("sb3_sorties", exist_ok=True)
    chemin_modele = "sb3_sorties/dqn_puissance4"

    # 1) Créer l'environnement et l'adversaire non-ML (Paul)
    env = creer_env()
    adversaire = AdversaireFactory.creer_adversaire_non_ml(Paul)
    env.set_adversaire(adversaire)

    # 2) Créer un nouveau modèle DQN (pas de chargement)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=200_000,
        learning_starts=1_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.30,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=0,
        seed=0,
    )

    # 3) Tester le résultat contre Paul
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"AgentML vs {adversaire.nom} avant apprentissage : "
        f"Victoires {100*taux_v:.1f} Défaites {100*taux_d:.1f} Nuls {100*taux_n:.1f} (sur 200)"
    )

    # 4) Apprendre contre Paul
    model.learn(total_timesteps=10_000, progress_bar=True)

    # 5) Tester le résultat contre Paul
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"AgentML vs {adversaire.nom} après apprentissage : "
        f"Victoires {100*taux_v:.1f} Défaites {100*taux_d:.1f} Nuls {100*taux_n:.1f} (sur 200)"
    )

    # 6) Sauvegarder le modèle
    model.save(chemin_modele)
    env.close()


if __name__ == "__main__":
    main()
