import argparse

from p4_tools import (
    Emmanuelle,
    Gilles,
    GrillePuissance4,
    Isabelle,
    JeanClaude,
    Nathalie,
    Nicolas,
    Paul,
    Pierre,
    Sylvain,
    Sylvie,
)


AGENT_FACTORIES = {
    "Paul": lambda env: Paul(),
    "Pierre": lambda env: Pierre(),
    "JeanClaude": lambda env: JeanClaude(env),
    "Nicolas": lambda env: Nicolas(env),
    "Isabelle": lambda env: Isabelle(env),
    "Sylvie": lambda env: Sylvie(env),
    "Gilles": lambda env: Gilles(),
    "Nathalie": lambda env: Nathalie(),
    "Emmanuelle": lambda env: Emmanuelle(env),
    "Sylvain": lambda env: Sylvain(env),
}


def jouer_une_partie(nom_j1, nom_j2):
    env = GrillePuissance4()
    env.reinitialiser()

    agent_j1 = AGENT_FACTORIES[nom_j1](env)
    agent_j2 = AGENT_FACTORIES[nom_j2](env)

    joueur = +1
    fini = False

    while not fini:
        if joueur == +1:
            agent_courant = agent_j1
        else:
            agent_courant = agent_j2

        actions = env.actions_valides()
        action = agent_courant.choisir_action(env.etat, actions, joueur)
        env.appliquer_action(action, joueur)

        fini, vainqueur = env.etat_terminal()
        joueur = -joueur

    if vainqueur == +1:
        return nom_j1
    if vainqueur == -1:
        return nom_j2
    return None


def initialiser_stats(noms_agents):
    stats = {}
    for nom in noms_agents:
        stats[nom] = {
            "jouees": 0,
            "gagnees": 0,
            "nulles": 0,
            "perdues": 0,
            "points": 0,
        }
    return stats


def mettre_a_jour_stats(stats, nom_j1, nom_j2, gagnant):
    stats[nom_j1]["jouees"] += 1
    stats[nom_j2]["jouees"] += 1

    if gagnant is None:
        stats[nom_j1]["nulles"] += 1
        stats[nom_j2]["nulles"] += 1
        stats[nom_j1]["points"] += 1
        stats[nom_j2]["points"] += 1
        return

    perdant = nom_j2 if gagnant == nom_j1 else nom_j1
    stats[gagnant]["gagnees"] += 1
    stats[perdant]["perdues"] += 1
    stats[gagnant]["points"] += 3


def _generer_une_ronde(noms, ronde_index):
    """
    Genere les duels d'une ronde via la methode du cercle (Berger).
    Chaque agent joue exactement une fois dans la ronde.
    """
    n = len(noms)
    assert n % 2 == 0, "Le nombre d'agents doit etre pair."

    rotation = noms[:]
    for _ in range(ronde_index):
        fixe = rotation[0]
        reste = rotation[1:]
        reste = [reste[-1]] + reste[:-1]
        rotation = [fixe] + reste

    duels = []
    for i in range(n // 2):
        a = rotation[i]
        b = rotation[n - 1 - i]

        # Alterne le joueur qui commence d'une ronde a l'autre.
        if ronde_index % 2 == 0:
            duels.append((a, b))
        else:
            duels.append((b, a))

    return duels


def jouer_tournoi(noms_agents, nb_parties_par_agent=100):
    if len(noms_agents) < 2:
        raise ValueError("Il faut au moins deux agents pour lancer un tournoi.")
    if len(noms_agents) % 2 != 0:
        raise ValueError(
            "Le nombre d'agents doit etre pair pour un planning equilibre."
        )
    if nb_parties_par_agent <= 0:
        raise ValueError("Le nombre de parties par agent doit etre > 0.")

    stats = initialiser_stats(noms_agents)
    nb_rondes = nb_parties_par_agent

    total_jouees = 0
    for ronde in range(nb_rondes):
        duels_ronde = _generer_une_ronde(noms_agents, ronde % (len(noms_agents) - 1))
        for nom_j1, nom_j2 in duels_ronde:
            gagnant = jouer_une_partie(nom_j1, nom_j2)
            mettre_a_jour_stats(stats, nom_j1, nom_j2, gagnant)
            total_jouees += 1

    return stats, total_jouees


def afficher_classement(stats):
    classement = sorted(
        stats.items(),
        key=lambda item: (item[1]["points"], item[1]["gagnees"], item[1]["nulles"]),
        reverse=True,
    )

    print("\n=== Classement des agents non-ML (tournoi) ===")
    print("Rang | Agent       | Pts | J | G | N | P")
    print("-----+-------------+-----+---+---+---+---")

    for rang, (nom, s) in enumerate(classement, start=1):
        print(
            f"{rang:>4} | {nom:<11} | {s['points']:>3} | {s['jouees']:>1} | "
            f"{s['gagnees']:>1} | {s['nulles']:>1} | {s['perdues']:>1}"
        )


def parser_args():
    parser = argparse.ArgumentParser(
        description="Lance un tournoi entre agents heuristiques de Puissance 4."
    )
    parser.add_argument(
        "--parties-par-agent",
        type=int,
        default=100,
        help="Nombre de parties jouees par chaque agent (defaut: 100).",
    )
    return parser.parse_args()


def main():
    args = parser_args()
    noms_agents = list(AGENT_FACTORIES.keys())
    parties_par_agent = args.parties_par_agent
    total_theorique = (len(noms_agents) * parties_par_agent) // 2

    stats, total_jouees = jouer_tournoi(
        noms_agents, nb_parties_par_agent=parties_par_agent
    )
    print(f"Agents                 : {len(noms_agents)}")
    print(f"Parties par agent      : {parties_par_agent}")
    print(f"Parties totales attendues: {total_theorique}")
    print(f"Parties jouees   : {total_jouees}")
    afficher_classement(stats)


if __name__ == "__main__":
    main()
