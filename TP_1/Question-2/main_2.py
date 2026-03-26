from __future__ import annotations

import os
from typing import Tuple

from stable_baselines3 import DQN

from p4_adversaires import AdversaireFactory
from p4_env_gymnasium import Puissance4GymnasiumEnv, creer_env
from p4_tools import Pierre


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
    # 1) Definir le chemin du modele deja entraine
    chemin_modele = "sb3_sorties/dqn_puissance4"

    # 2) Verifier que le modele existe avant de continuer
    if not os.path.exists(chemin_modele + ".zip"):
        raise FileNotFoundError(
            f"Modele introuvable: {chemin_modele}.zip. Lance d'abord main_1.py."
        )

    # 3) Creer l'environnement et configurer l'adversaire non-ML (Pierre)
    env = creer_env(seed=0)
    adversaire = AdversaireFactory.creer_adversaire_non_ml(Pierre)
    env.set_adversaire(adversaire)

    # 4) Charger le modele sauvegarde
    model = DQN.load(chemin_modele, env=env)

    # 5) Evaluer les performances avant nouvel apprentissage
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"AgentML vs {adversaire.nom} avant apprentissage : "
        f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
    )

    # 6) Continuer l'apprentissage contre Pierre
    model.learn(total_timesteps=10_000, reset_num_timesteps=False, progress_bar=True)

    # 7) Evaluer a nouveau apres apprentissage
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"AgentML vs {adversaire.nom} apres apprentissage : "
        f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
    )

    # 8) Sauvegarder le modele mis a jour et fermer l'environnement
    model.save(chemin_modele)
    env.close()


if __name__ == "__main__":
    main()
