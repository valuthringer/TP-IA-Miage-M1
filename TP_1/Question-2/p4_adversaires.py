from __future__ import annotations

import os
import numpy as np
from stable_baselines3 import DQN


class AgentSnapshotModele:
    def __init__(self, modele_snapshot, nom: str = "SnapshotDQN") -> None:
        self.modele_snapshot = modele_snapshot
        self.nom = nom

    def choisir_action(self, etat, actions_valides, joueur):
        obs = np.asarray(etat, dtype=np.int8).reshape(42)
        if joueur == -1:
            obs = -obs

        action, _ = self.modele_snapshot.predict(obs, deterministic=True)
        action = int(action)
        if action not in actions_valides:
            action = int(actions_valides[0])
        return action


class AdversaireFactory:
    @staticmethod
    def creer_adversaire_non_ml(classe_agent_non_ml):
        return classe_agent_non_ml()

    @staticmethod
    def creer_adversaire_snapchot(chemin_modele_zip):
        if not isinstance(chemin_modele_zip, str) or not chemin_modele_zip.endswith(".zip"):
            raise ValueError("chemin_modele_zip doit inclure explicitement l'extension .zip")
        modele_snapshot = DQN.load(chemin_modele_zip)
        nom_snapshot = os.path.splitext(os.path.basename(chemin_modele_zip))[0]
        return AgentSnapshotModele(modele_snapshot, nom=nom_snapshot)

    @staticmethod
    def creer_adversaire_model(modele):
        return AgentSnapshotModele(modele)

    @staticmethod
    def creer_adversaire_snapshot_live(modele_courant, chemin_snapshot_base):
        modele_courant.save(chemin_snapshot_base)
        modele_snapshot = DQN.load(chemin_snapshot_base)
        return AgentSnapshotModele(modele_snapshot)

    @staticmethod
    def depuis_spec(spec, modele_courant, chemin_snapshot_base):
        type_adversaire = spec["type"]
        source = spec["source"]

        if type_adversaire == "non_ml":
            return AdversaireFactory.creer_adversaire_non_ml(source)
        if type_adversaire == "snapshot_file":
            return AdversaireFactory.creer_adversaire_snapchot(source)
        if type_adversaire == "snapshot_live":
            if modele_courant is None:
                raise ValueError("snapshot_live requiert un modèle courant déjà initialisé")
            return AdversaireFactory.creer_adversaire_snapshot_live(modele_courant, chemin_snapshot_base)

        raise ValueError(f"type d'adversaire inconnu: {type_adversaire}")
