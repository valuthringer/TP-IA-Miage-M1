from __future__ import annotations

import os
from typing import Tuple

from stable_baselines3 import DQN

from p4_adversaires import AdversaireFactory
from p4_env_gymnasium import Puissance4GymnasiumEnv, creer_env


def evaluer_modele(model, env: Puissance4GymnasiumEnv, nb_parties: int) -> Tuple[float, float, float]:
    vic, defa, nul = 0, 0, 0

    for _ in range(nb_parties):
        obs, info = env.reset()
        termine = False
        while not termine:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
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

    # 1) Modele deja sauvegarde qui servira d'adversaire (fige)
    chemin_modele_adversaire = "sb3_sorties/dqn_puissance4.zip"
    if not os.path.exists(chemin_modele_adversaire):
        raise FileNotFoundError(
            f"Modele adversaire introuvable: {chemin_modele_adversaire}. Lance d'abord main_1.py."
        )

    # 2) Chemin de sauvegarde du NOUVEL agent que l'on entraine ici
    chemin_nouvel_agent = "sb3_sorties/dqn_puissance4_v2"

    # 3) Creer l'adversaire a partir du modele sauvegarde
    adversaire = AdversaireFactory.creer_adversaire_snapchot(chemin_modele_adversaire)

    # 4) Creer l'environnement et y injecter cet adversaire
    env = creer_env(seed=0)
    env.set_adversaire(adversaire)

    # 5) Creer un NOUVEAU modele DQN (depuis zero)
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=200_000,
        learning_starts=10_000,
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

    # 6) Evaluer avant apprentissage
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"NouvelAgent vs {adversaire.nom} avant apprentissage : "
        f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
    )

    # 7) Entrainer le nouvel agent contre l'adversaire fige
    model.learn(total_timesteps=20_000, progress_bar=True)

    # 8) Evaluer apres apprentissage
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"NouvelAgent vs {adversaire.nom} apres apprentissage : "
        f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
    )

    # 9) Sauvegarder le nouvel agent
    model.save(chemin_nouvel_agent)
    env.close()


if __name__ == "__main__":
    main()
