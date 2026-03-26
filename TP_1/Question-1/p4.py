import random

from p4_tools import (
    GrillePuissance4,
    Paul,
    Pierre,
    JeanClaude,
    Nicolas,
    Isabelle,
    Sylvie,
    Gilles,
    Nathalie,
    Emmanuelle,
    Sylvain
)


def jouer_une_partie(environnement, agent_j1, agent_j2):
    etat = environnement.reinitialiser()
    joueur = +1
    fini = False

    while not fini :

        if joueur == +1 :
            agent_courant = agent_j1
        else:
            agent_courant = agent_j2
        
        actions = environnement.actions_valides()
        
        action = agent_courant.choisir_action(etat, actions, joueur)
        
        environnement.appliquer_action(action, joueur)

        fini, vainqueur = environnement.etat_terminal()

        joueur = -joueur

    return vainqueur


def main():
    env = GrillePuissance4()

    agent_1 = Pierre()
    agent_2 = Paul()

    vainqueur = jouer_une_partie(env, agent_1, agent_2)

    if vainqueur == +1:
        print(f"Vainqueur : {agent_1.nom}")
    elif vainqueur == -1:
        print(f"Vainqueur : {agent_2.nom}")
    else:
        print("Résultat : match nul")


if __name__ == "__main__":
    main()
