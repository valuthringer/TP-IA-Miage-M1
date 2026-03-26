from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from p4_adversaires import AdversaireFactory
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
    GrillePuissance4,
)

NB_PARTIES_PAR_DUEL = 10


@dataclass
class FicheAgent:
    nom: str
    agent: object
    victoires: int = 0
    defaites: int = 0
    nuls: int = 0


def jouer_une_partie(agent_a, agent_b, agent_a_commence: bool) -> int:
    """
    Retourne:
    - 1 si agent_a gagne
    - -1 si agent_b gagne
    - 0 si nul
    """
    grille = GrillePuissance4()

    if agent_a_commence:
        agent_plus, agent_moins = agent_a, agent_b
    else:
        agent_plus, agent_moins = agent_b, agent_a

    joueur_courant = +1

    while True:
        actions = grille.actions_valides()
        if joueur_courant == +1:
            action = int(agent_plus.choisir_action(grille.etat, actions, +1))
        else:
            action = int(agent_moins.choisir_action(grille.etat, actions, -1))

        if action not in actions:
            action = int(actions[0])

        grille.appliquer_action(action, joueur_courant)

        termine, vainqueur = grille.etat_terminal()
        if termine:
            if vainqueur == 0:
                return 0
            if agent_a_commence:
                return 1 if vainqueur == +1 else -1
            return 1 if vainqueur == -1 else -1

        joueur_courant *= -1


def jouer_duel(fiche_a: FicheAgent, fiche_b: FicheAgent, nb_parties: int) -> tuple[int, int, int]:
    vic_a, vic_b, nuls = 0, 0, 0

    for i in range(nb_parties):
        a_commence = (i % 2 == 0)
        resultat = jouer_une_partie(fiche_a.agent, fiche_b.agent, a_commence)

        if resultat == 1:
            vic_a += 1
            fiche_a.victoires += 1
            fiche_b.defaites += 1
        elif resultat == -1:
            vic_b += 1
            fiche_b.victoires += 1
            fiche_a.defaites += 1
        else:
            nuls += 1
            fiche_a.nuls += 1
            fiche_b.nuls += 1

    return vic_a, vic_b, nuls


def creer_liste_agents(chemin_modele_zip: str) -> List[FicheAgent]:
    agents_non_ml = [
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

    fiches = [FicheAgent(nom=cls.nom, agent=AdversaireFactory.creer_adversaire_non_ml(cls)) for cls in agents_non_ml]

    agent_modele = AdversaireFactory.creer_adversaire_snapchot(chemin_modele_zip)
    fiches.append(FicheAgent(nom=agent_modele.nom, agent=agent_modele))
    return fiches


def afficher_classement(fiches: List[FicheAgent]) -> None:
    classement = sorted(
        fiches,
        key=lambda f: (f.victoires, -f.defaites),
        reverse=True,
    )

    print("\nClassement final (ordre decroissant de victoires)")
    for i, f in enumerate(classement, start=1):
        print(
            f"{i:2d}. {f.nom:20s} "
            f"V={f.victoires:4d} D={f.defaites:4d} N={f.nuls:4d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Championnat tous-contre-tous entre agents non-ML et un modele charge."
    )
    parser.add_argument(
        "chemin_modele_zip",
        type=str,
        help="Chemin du modele SB3 a charger (avec extension .zip).",
    )
    args = parser.parse_args()

    fiches = creer_liste_agents(args.chemin_modele_zip)

    for i in range(len(fiches)):
        for j in range(i + 1, len(fiches)):
            vic_i, vic_j, nuls = jouer_duel(fiches[i], fiches[j], NB_PARTIES_PAR_DUEL)
            print(
                f"Duel: {fiches[i].nom} vs {fiches[j].nom} | "
                f"{fiches[i].nom}: V={vic_i} D={vic_j} N={nuls}"
            )

    afficher_classement(fiches)


if __name__ == "__main__":
    main()
