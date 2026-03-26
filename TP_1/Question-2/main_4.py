from __future__ import annotations

import os
from typing import Tuple

from stable_baselines3 import DQN

from p4_adversaires import AdversaireFactory
from p4_env_gymnasium import Puissance4GymnasiumEnv, creer_env
from p4_tools import Nicolas, Sylvain


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
    chemin_modele = "sb3_sorties/dqn_puissance4_v2"
    if not os.path.exists(chemin_modele + ".zip"):
        raise FileNotFoundError(f"Modele introuvable: {chemin_modele}.zip")

    env = creer_env(seed=0)
    model = DQN.load(chemin_modele, env=env)

    # 1) Test contre Nicolas
    adversaire_nicolas = AdversaireFactory.creer_adversaire_non_ml(Nicolas)
    env.set_adversaire(adversaire_nicolas)
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"AgentML vs {adversaire_nicolas.nom} avant self-play : "
        f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
    )

    # 2) Apprentissage contre lui-meme (self-play dynamique)
    adversaire_self = AdversaireFactory.creer_adversaire_model(model)
    env.set_adversaire(adversaire_self)
    model.learn(total_timesteps=50_000, reset_num_timesteps=False, progress_bar=True)

    # 3) Re-test contre Sylvain
    adversaire_sylvain = AdversaireFactory.creer_adversaire_non_ml(Nicolas)
    env.set_adversaire(adversaire_sylvain)
    taux_v, taux_d, taux_n = evaluer_modele(model, env, 200)
    print(
        f"AgentML vs {adversaire_sylvain.nom} apres self-play : "
        f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
    )

    model.save(chemin_modele)
    env.close()


if __name__ == "__main__":
    main()
