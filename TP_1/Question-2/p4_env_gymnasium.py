# p4_env_gymnasium.py
# Environnement Gymnasium (compatible SB3) pour Puissance 4,
# basé directement sur GrillePuissance4 (p4_tools.py).

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from p4_tools import GrillePuissance4, Paul


class Puissance4GymnasiumEnv(gym.Env):
    """Environnement Gymnasium : obs=Box(42,), action=Discrete(7)."""

    def __init__(
        self,
        penalite_action_invalide: float = -0.2,
        remplacer_action_invalide_par_aleatoire: bool = True,
        adversaire=None,
        adversaire_commence_aleatoire: bool = False,
    ) -> None:
        super().__init__()

        self.penalite_action_invalide = float(penalite_action_invalide)
        self._rng = np.random.default_rng()
        self._remplacer_aleatoire = bool(remplacer_action_invalide_par_aleatoire)
        self._adversaire_commence_aleatoire = bool(adversaire_commence_aleatoire)
        self.grille = GrillePuissance4()
        self.adversaire = adversaire if adversaire is not None else Paul()

        self.termine = False
        self.vainqueur = None

        # Observation : vecteur de 42 valeurs dans {-1,0,1}
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(42,),
            dtype=np.int8,
        )

        # Action : colonne 0..6
        self.action_space = spaces.Discrete(7)

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def set_adversaire(self, adversaire) -> None:
        self.adversaire = adversaire

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.grille.reinitialiser()
        self.termine = False
        self.vainqueur = None

        info: Dict[str, Any] = {"actions_valides": self.grille.actions_valides()}

        adversaire_commence = False
        if options is not None and "adversaire_commence" in options:
            adversaire_commence = bool(options["adversaire_commence"])
        elif self._adversaire_commence_aleatoire:
            adversaire_commence = bool(self._rng.integers(0, 2))

        if adversaire_commence:
            actions_adv = self.grille.actions_valides()
            action_adv = int(self.adversaire.choisir_action(self.grille.etat, actions_adv, joueur=-1))
            if action_adv not in actions_adv:
                action_adv = int(actions_adv[0])
            self.grille.appliquer_action(action_adv, joueur=-1)
            info["action_adversaire_depart"] = action_adv
            info["actions_valides"] = self.grille.actions_valides()

        info["adversaire_commence"] = adversaire_commence
        obs = self._observation()
        return obs, info

    def step(self, action: int):
        # Gymnasium fournit parfois np.int64
        action_int = int(action)
        info: Dict[str, Any] = {}

        if self.termine:
            return self._observation(), 0.0, True, False, {"deja_termine": True}

        actions_valides = self.grille.actions_valides()
        info["actions_valides"] = actions_valides
        reward = 0.0

        if action_int not in actions_valides:

            info["action_invalide"] = True
            info["action_demandee"] = action_int

            # pénalité forte
            reward = -1.0

            # épisode terminé
            self.termine = True
            self.vainqueur = -1

            info["vainqueur"] = -1
            info["raison"] = "action_invalide"

            return self._observation(), float(reward), True, False, info

        # Coup de l'agent apprenant (+1)
        self.grille.appliquer_action(action_int, joueur=+1)
        fini, vainqueur = self.grille.etat_terminal()
        if fini:
            self.termine = True
            self.vainqueur = vainqueur
            reward += self._recompense_finale(vainqueur)
            info["vainqueur"] = vainqueur
            return self._observation(), float(reward), True, False, info

        # Coup de l'adversaire (-1)
        etat = self.grille.etat
        actions_adv = self.grille.actions_valides()
        action_adv = self.adversaire.choisir_action(etat, actions_adv, joueur=-1)
        self.grille.appliquer_action(action_adv, joueur=-1)
        info["action_adversaire"] = int(action_adv)

        fini, vainqueur = self.grille.etat_terminal()
        if fini:
            self.termine = True
            self.vainqueur = vainqueur
            reward += self._recompense_finale(vainqueur)
            info["vainqueur"] = vainqueur
            return self._observation(), float(reward), True, False, info

        info["actions_valides"] = self.grille.actions_valides()
        return self._observation(), float(reward), False, False, info

    @staticmethod
    def _aplatir_grille(grille) -> list[int]:
        return [v for ligne in grille for v in ligne]

    def _observation(self) -> np.ndarray:
        return np.asarray(self._aplatir_grille(self.grille.etat), dtype=np.int8)

    @staticmethod
    def _recompense_finale(vainqueur: Optional[int]) -> float:
        if vainqueur is None:
            return 0.0
        if vainqueur == 0:
            return 0.0
        if vainqueur == +1:
            return 1.0
        return -1.0


def creer_env(
    penalite_action_invalide: float = -0.2,
    seed: int = 0,
    adversaire=None,
    adversaire_commence_aleatoire: bool = False,
) -> Puissance4GymnasiumEnv:
    env = Puissance4GymnasiumEnv(
        penalite_action_invalide=penalite_action_invalide,
        remplacer_action_invalide_par_aleatoire=True,
        adversaire_commence_aleatoire=adversaire_commence_aleatoire,
    )
    if adversaire is not None:
        env.set_adversaire(adversaire)
    env.reset(seed=seed)
    return env
