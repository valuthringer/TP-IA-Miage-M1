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
    dossier_snapshots = "sb3_sorties/snapshots_selfplay"
    os.makedirs(dossier_snapshots, exist_ok=True)

    chemin_modele = "sb3_sorties/dqn_puissance4_v2"
    nb_blocs = 3
    timesteps_par_bloc = 10_000
    nb_parties_eval = 100

    # 1) Environnement principal
    env = creer_env(seed=0)

    # 2) Charger le modèle s'il existe, sinon créer un nouveau modèle
    if os.path.exists(chemin_modele + ".zip"):
        model = DQN.load(chemin_modele, env=env)
    else:
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

    # 3) Premier adversaire: le modèle courant (self-play dynamique)
    adversaire = AdversaireFactory.creer_adversaire_model(model)
    env.set_adversaire(adversaire)

    for bloc in range(1, nb_blocs + 1):
        # 4) Apprentissage contre l'adversaire courant
        model.learn(total_timesteps=timesteps_par_bloc, reset_num_timesteps=False, progress_bar=True)

        # 5) Test contre ce même adversaire
        env_eval = creer_env(seed=0, adversaire=adversaire)
        taux_v, taux_d, taux_n = evaluer_modele(model, env_eval, nb_parties_eval)
        env_eval.close()
        print(
            f"[BLOC {bloc}] vs {adversaire.nom} | "
            f"V={100*taux_v:.1f}% D={100*taux_d:.1f}% N={100*taux_n:.1f}%"
        )

        # 6) Sauvegarder un snapshot du modèle courant
        chemin_snapshot_base = os.path.join(dossier_snapshots, f"snapshot_{bloc:03d}")
        model.save(chemin_snapshot_base)
        chemin_snapshot_zip = chemin_snapshot_base + ".zip"

        # 7) Recharger ce snapshot comme nouvel adversaire pour le bloc suivant
        adversaire = AdversaireFactory.creer_adversaire_snapchot(chemin_snapshot_zip)
        env.set_adversaire(adversaire)

    # 8) Sauvegarder le modèle final
    model.save(chemin_modele)


if __name__ == "__main__":
    main()
