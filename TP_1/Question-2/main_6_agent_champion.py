from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from stable_baselines3 import DQN

from p4_adversaires import AdversaireFactory
from p4_env_gymnasium import Puissance4GymnasiumEnv, creer_env
from p4_tools import (
    Emmanuelle,
    Gilles,
    Isabelle,
    JeanClaude,
    Nathalie,
    Nicolas,
    Paul,
    Pierre,
    Sylvain,
    Sylvie,
)


def _progress_bar_disponible() -> bool:
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def apprendre_modele(
    model: DQN, total_timesteps: int, reset_num_timesteps: bool
) -> None:
    """Lance l'apprentissage sans planter si tqdm/rich ne sont pas installes."""
    use_progress_bar = _progress_bar_disponible()
    if not use_progress_bar:
        print("Info: tqdm/rich non installes, apprentissage sans barre de progression.")

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=use_progress_bar,
    )


@dataclass
class Bilan:
    victoires: int = 0
    defaites: int = 0
    nuls: int = 0

    @property
    def winrate(self) -> float:
        total = self.victoires + self.defaites + self.nuls
        if total == 0:
            return 0.0
        return self.victoires / total


def evaluer_vs_adversaire(
    model: DQN, env: Puissance4GymnasiumEnv, nb_parties: int
) -> Bilan:
    bilan = Bilan()

    for _ in range(nb_parties):
        obs, _ = env.reset()
        termine = False

        while not termine:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            termine = bool(terminated or truncated)

        vainqueur = info.get("vainqueur", None)
        if vainqueur == +1:
            bilan.victoires += 1
        elif vainqueur == -1:
            bilan.defaites += 1
        else:
            bilan.nuls += 1

    return bilan


def evaluer_vs_tous_les_non_ml(
    model: DQN, nb_parties_par_adversaire: int = 40
) -> Tuple[float, Dict[str, Bilan]]:
    classes_non_ml = [
        Paul,
        Pierre,
        JeanClaude,
        Nicolas,
        Sylvie,
        Isabelle,
        Gilles,
        Nathalie,
        Emmanuelle,
        Sylvain,
    ]

    resultats: Dict[str, Bilan] = {}
    total_v, total_d, total_n = 0, 0, 0

    for cls in classes_non_ml:
        adversaire = AdversaireFactory.creer_adversaire_non_ml(cls)
        env_eval = creer_env(
            seed=0, adversaire=adversaire, adversaire_commence_aleatoire=True
        )
        bilan = evaluer_vs_adversaire(model, env_eval, nb_parties_par_adversaire)
        env_eval.close()

        resultats[cls.nom] = bilan
        total_v += bilan.victoires
        total_d += bilan.defaites
        total_n += bilan.nuls

    total = max(1, total_v + total_d + total_n)
    score_global = (3 * total_v + total_n) / (3 * total)
    return score_global, resultats


def afficher_resultats(resultats: Dict[str, Bilan]) -> None:
    print("\n--- Evaluation vs agents non-ML ---")
    for nom in sorted(resultats.keys()):
        b = resultats[nom]
        total = b.victoires + b.defaites + b.nuls
        print(
            f"{nom:12s} | V={b.victoires:3d} D={b.defaites:3d} N={b.nuls:3d} "
            f"| WR={100.0*b.winrate:5.1f}% | n={total:3d}"
        )


def construire_modele(env: Puissance4GymnasiumEnv, seed: int) -> DQN:
    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.5e-4,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.03,
        verbose=0,
        seed=seed,
    )


def main() -> None:
    os.makedirs("sb3_sorties", exist_ok=True)
    os.makedirs("sb3_sorties/snapshots_champion", exist_ok=True)

    chemin_base = "sb3_sorties/dqn_puissance4_champion"
    chemin_best = "sb3_sorties/dqn_puissance4_champion_best"

    seed = 0
    env = creer_env(seed=seed, adversaire_commence_aleatoire=True)

    if os.path.exists(chemin_base + ".zip"):
        print("Chargement du modele existant...")
        model = DQN.load(chemin_base, env=env)
    else:
        print("Creation d'un nouveau modele...")
        model = construire_modele(env, seed=seed)

    # 1) Entrainement melange (shuffle) contre tous les non-ML
    adversaires_non_ml: List[type] = [
        Paul,
        Pierre,
        JeanClaude,
        Nicolas,
        Sylvie,
        Isabelle,
        Gilles,
        Nathalie,
        Emmanuelle,
        Sylvain,
    ]

    # 6 passages * 10 adversaires * 4_500 = 270_000 steps (proche de l'ancien budget)
    nb_passages_shuffle = 6
    timesteps_par_bloc = 4_500
    rng = random.Random(seed)

    meilleur_score = -1.0

    for passage in range(1, nb_passages_shuffle + 1):
        ordre = adversaires_non_ml[:]
        rng.shuffle(ordre)
        ordre_noms = ", ".join(cls.nom for cls in ordre)
        print(
            f"\n[SHUFFLE] Passage {passage}/{nb_passages_shuffle} | ordre: {ordre_noms}"
        )

        for cls_adv in ordre:
            adversaire = AdversaireFactory.creer_adversaire_non_ml(cls_adv)
            env.set_adversaire(adversaire)
            print(
                f"[SHUFFLE] Entrainement vs {cls_adv.nom} "
                f"({timesteps_par_bloc} steps)"
            )
            apprendre_modele(
                model,
                total_timesteps=timesteps_par_bloc,
                reset_num_timesteps=False,
            )

        score, details = evaluer_vs_tous_les_non_ml(model, nb_parties_par_adversaire=30)
        print(f"Score global apres passage shuffle {passage}: {100.0*score:.2f}%")
        afficher_resultats(details)

        if score > meilleur_score:
            meilleur_score = score
            model.save(chemin_best)
            print(f"Nouveau meilleur modele sauvegarde: {chemin_best}.zip")

        model.save(chemin_base)

    # 2) Self-play avec snapshots figes
    print("\n[SELF-PLAY] Debut des cycles de self-play...")
    nb_cycles = 5
    timesteps_par_cycle = 30_000

    for cycle in range(1, nb_cycles + 1):
        chemin_snapshot = f"sb3_sorties/snapshots_champion/snapshot_cycle_{cycle:03d}"
        model.save(chemin_snapshot)

        adversaire_snapshot = AdversaireFactory.creer_adversaire_snapchot(
            chemin_snapshot + ".zip"
        )
        env.set_adversaire(adversaire_snapshot)

        print(f"[SELF-PLAY] Cycle {cycle}/{nb_cycles} ({timesteps_par_cycle} steps)")
        apprendre_modele(
            model,
            total_timesteps=timesteps_par_cycle,
            reset_num_timesteps=False,
        )

        score, details = evaluer_vs_tous_les_non_ml(model, nb_parties_par_adversaire=40)
        print(f"Score global apres self-play cycle {cycle}: {100.0*score:.2f}%")
        afficher_resultats(details)

        if score > meilleur_score:
            meilleur_score = score
            model.save(chemin_best)
            print(f"Nouveau meilleur modele sauvegarde: {chemin_best}.zip")

        model.save(chemin_base)

    env.close()
    print("\nTermine.")
    print(f"Modele courant: {chemin_base}.zip")
    print(f"Meilleur modele: {chemin_best}.zip")


if __name__ == "__main__":
    main()
