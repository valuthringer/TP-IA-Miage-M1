
import random
from typing import List, Optional, Tuple

class GrillePuissance4:
    NB_LIGNES = 6
    NB_COLONNES = 7

    def __init__(self, autre=None):
        if autre is None:
            self.etat = None
            self.reinitialiser()
        elif isinstance(autre, GrillePuissance4):
            self.etat = [ligne[:] for ligne in autre.etat]
        else:
            raise TypeError("autre doit etre un GrillePuissance4 ou None.")

    def reinitialiser(self):
        self.etat = [[0 for _ in range(self.NB_COLONNES)] for _ in range(self.NB_LIGNES)]

    def actions_valides(self):
        return [c for c in range(self.NB_COLONNES) if self.etat[0][c] == 0]

    def appliquer_action(self, action, joueur):
        nouvel_etat = [ligne[:] for ligne in self.etat]
        for r in range(self.NB_LIGNES - 1, -1, -1):
            if nouvel_etat[r][action] == 0:
                nouvel_etat[r][action] = joueur
                self.etat = nouvel_etat
                return
        raise ValueError("Action invalide : colonne pleine")

    def etat_terminal(self):
        if self._a_4_alignes(+1):
            return True, +1
        if self._a_4_alignes(-1):
            return True, -1
        if not self.actions_valides():
            return True, 0
        return False, None

    def _a_4_alignes(self, joueur):
        etat = self.etat
        r_max, c_max = self.NB_LIGNES, self.NB_COLONNES

        for r in range(r_max):
            for c in range(c_max - 3):
                if all(etat[r][c + i] == joueur for i in range(4)):
                    return True

        for r in range(r_max - 3):
            for c in range(c_max):
                if all(etat[r + i][c] == joueur for i in range(4)):
                    return True

        for r in range(r_max - 3):
            for c in range(c_max - 3):
                if all(etat[r + i][c + i] == joueur for i in range(4)):
                    return True

        for r in range(3, r_max):
            for c in range(c_max - 3):
                if all(etat[r - i][c + i] == joueur for i in range(4)):
                    return True

        return False


def grille_depuis_etat(etat):
    grille = GrillePuissance4()
    grille.etat = [ligne[:] for ligne in etat]
    return grille

class AgentBase:
    nom = "AgentBase"

    def __init__(self) -> None:
        pass

    def choisir_action(self, etat, actions_valides, joueur):
        raise NotImplementedError

    # Hooks optionnels (ignorés par agents non-ML)
    def enregistrer_transition(self, etat, action, recompense, etat_suivant, partie_terminee, joueur):
        return

    def apprendre(self):
        return

class Paul(AgentBase):
    nom = "Paul"

    def choisir_action(self, etat, actions_valides, joueur):
        return random.choice(actions_valides)

class Pierre(AgentBase):
    nom = "Pierre"

    def choisir_action(self, etat, actions_valides, joueur):
        if 3 in actions_valides:
            return 3
        return random.choice(actions_valides)

class JeanClaude(AgentBase):
    nom = "JeanClaude"

    def choisir_action(self, etat, actions_valides, joueur):
        base = grille_depuis_etat(etat)
        for a in actions_valides:
            plateau2 = GrillePuissance4(base)
            plateau2.appliquer_action(a, joueur)
            fini, vainqueur = plateau2.etat_terminal()
            if fini and vainqueur == joueur:
                return a
        if 3 in actions_valides:
            return 3
        return random.choice(actions_valides)
    
class Nicolas(AgentBase):
    nom = "Nicolas"

    def choisir_action(self, etat, actions_valides, joueur):
        adversaire = -joueur
        base = grille_depuis_etat(etat)
        for a in actions_valides:
            plateau2 = GrillePuissance4(base)
            plateau2.appliquer_action(a, adversaire)
            fini, vainqueur = plateau2.etat_terminal()
            if fini and vainqueur == adversaire:
                return a
        if 3 in actions_valides:
            return 3
        return random.choice(actions_valides)

class Sylvie(AgentBase):
    nom = "Sylvie"

    def choisir_action(self, etat, actions_valides, joueur):
        meilleur_score = None
        meilleure_action = None
        base = grille_depuis_etat(etat)
        for a in actions_valides:
            plateau1 = GrillePuissance4(base)
            plateau1.appliquer_action(a, joueur)
            fini, vainqueur = plateau1.etat_terminal()
            if fini and vainqueur == joueur:
                return a  # gagne direct

            actions_adv = plateau1.actions_valides()
            adversaire = -joueur
            danger = False
            for a2 in actions_adv:
                plateau2 = GrillePuissance4(plateau1)
                plateau2.appliquer_action(a2, adversaire)
                fini2, vainqueur2 = plateau2.etat_terminal()
                if fini2 and vainqueur2 == adversaire:
                    danger = True
                    break

            score = 0
            if danger:
                score -= 100

            if a == 3:
                score += 3
            elif a in [2, 4]:
                score += 2
            elif a in [1, 5]:
                score += 1

            if meilleur_score is None or score > meilleur_score:
                meilleur_score = score
                meilleure_action = a

        if meilleure_action is None:
            return random.choice(actions_valides)
        return meilleure_action

class Isabelle(AgentBase):
    nom = "Isabelle"

    def choisir_action(self, etat, actions_valides, joueur):
        # 1) Si je peux gagner en 1 coup
        base = grille_depuis_etat(etat)
        for a in actions_valides:
            plateau2 = GrillePuissance4(base)
            plateau2.appliquer_action(a, joueur)
            fini, vainqueur = plateau2.etat_terminal()
            if fini and vainqueur == joueur:
                return a

        # 2) Sinon, si l'adversaire peut gagner en 1 coup, je bloque
        adversaire = -joueur
        for a in actions_valides:
            plateau2 = GrillePuissance4(base)
            plateau2.appliquer_action(a, adversaire)
            fini, vainqueur = plateau2.etat_terminal()
            if fini and vainqueur == adversaire:
                return a

        # 3) Sinon centre si possible, sinon random
        if 3 in actions_valides:
            return 3
        return random.choice(actions_valides)

class Gilles(AgentBase):
    nom = "Gilles"

    def choisir_action(self, etat, actions_valides, joueur):
        ordre = [3, 2, 4, 1, 5, 0, 6]
        for c in ordre:
            if c in actions_valides:
                return c
        return random.choice(actions_valides)
    
class Nathalie(AgentBase):
    nom = "Nathalie"

    def choisir_action(self, etat, actions_valides, joueur):
        groupes = [[3], [2, 4], [1, 5], [0, 6]]
        for g in groupes:
            candidats = []
            for c in g:
                if c in actions_valides:
                    candidats.append(c)
            if len(candidats) > 0:
                return random.choice(candidats)
        return random.choice(actions_valides)
    
class Emmanuelle(AgentBase):

    nom = "Emmanuelle"

    def __init__(self) -> None:
        super().__init__()
        self.ordre_colonnes = [3, 2, 4, 1, 5, 0, 6]

    def choisir_action(self, etat, actions_valides, joueur):
        adversaire = -joueur

        meilleure_action = None
        meilleur_score = None
        base = grille_depuis_etat(etat)

        # On évalue chaque action du joueur courant
        for action in actions_valides:
            plateau_apres_moi = GrillePuissance4(base)
            plateau_apres_moi.appliquer_action(action, joueur)

            partie_terminee, vainqueur = plateau_apres_moi.etat_terminal()
            if partie_terminee:
                # si je gagne tout de suite, c'est optimal
                if vainqueur == joueur:
                    return action
                # si c'est nul tout de suite, on lui donne un bon score mais pas infini
                if vainqueur == 0:
                    score_action = 0.0
                else:
                    # perdre sur mon propre coup n'arrive normalement pas, mais on garde le cas
                    score_action = -1.0
            else:
                # Réponse optimale de l'adversaire : il choisit l'action qui minimise mon score
                actions_adversaire = plateau_apres_moi.actions_valides()

                pire_score_pour_moi = None
                for action_adv in actions_adversaire:
                    plateau_apres_lui = GrillePuissance4(plateau_apres_moi)
                    plateau_apres_lui.appliquer_action(action_adv, adversaire)

                    term2, vainqueur2 = plateau_apres_lui.etat_terminal()
                    if term2:
                        if vainqueur2 == adversaire:
                            score_etat = -1.0  # très mauvais pour moi
                        elif vainqueur2 == joueur:
                            score_etat = 1.0   # très bon pour moi (rare après coup adverse)
                        else:
                            score_etat = 0.0
                    else:
                        score_etat = self._heuristique(plateau_apres_lui.etat, joueur)

                    if pire_score_pour_moi is None:
                        pire_score_pour_moi = score_etat
                    else:
                        if score_etat < pire_score_pour_moi:
                            pire_score_pour_moi = score_etat

                score_action = pire_score_pour_moi

            # Comparaison et tie-break (centre)
            if meilleur_score is None:
                meilleur_score = score_action
                meilleure_action = action
            else:
                if score_action > meilleur_score:
                    meilleur_score = score_action
                    meilleure_action = action
                elif score_action == meilleur_score:
                    # tie-break : colonne plus proche du centre
                    if self._est_preferable(action, meilleure_action):
                        meilleure_action = action

        # sécurité
        if meilleure_action is None:
            return random.choice(actions_valides)

        return meilleure_action

    def _est_preferable(self, action, action_actuelle) -> bool:
        # renvoie True si action est "plus proche du centre" que action_actuelle
        rang = self.ordre_colonnes.index(action)
        rang_actuel = self.ordre_colonnes.index(action_actuelle)
        return rang < rang_actuel

    def _heuristique(self, etat, joueur: int) -> float:
        """
        Heuristique simple :
        - bonus centre
        - bonus pour alignements 2 et 3
        - malus pour alignements adverses 2 et 3
        Valeur dans environ [-1, 1] (approximatif), suffisante pour départager.
        """
        adversaire = -joueur

        score = 0.0

        # (A) bonus centre
        score += 0.02 * self._compter_pions_centre(etat, joueur)
        score -= 0.02 * self._compter_pions_centre(etat, adversaire)

        # (B) alignements
        score += 0.03 * self._compter_alignements(etat, joueur, longueur=2)
        score += 0.08 * self._compter_alignements(etat, joueur, longueur=3)

        score -= 0.03 * self._compter_alignements(etat, adversaire, longueur=2)
        score -= 0.10 * self._compter_alignements(etat, adversaire, longueur=3)

        # clamp léger (pas indispensable, mais évite des valeurs trop grandes)
        if score > 1.0:
            score = 1.0
        if score < -1.0:
            score = -1.0

        return score

    def _compter_pions_centre(self, etat, joueur: int) -> int:
        # colonne centrale = 3
        c = 3
        nb = 0
        for r in range(GrillePuissance4.NB_LIGNES):
            if etat[r][c] == joueur:
                nb += 1
        return nb

    def _compter_alignements(self, etat, joueur: int, longueur: int) -> int:
        """
        Compte le nombre de "fenêtres de 4" qui contiennent exactement `longueur` pions du joueur
        et 0 pion adverse (les autres cases sont vides).
        C'est une heuristique standard/simple.
        """
        adversaire = -joueur
        total = 0

        R = GrillePuissance4.NB_LIGNES
        C = GrillePuissance4.NB_COLONNES

        # liste des lignes de 4 cases à vérifier
        fenetres = []

        # horizontales
        for r in range(R):
            for c in range(C - 3):
                fenetres.append([(r, c), (r, c + 1), (r, c + 2), (r, c + 3)])

        # verticales
        for r in range(R - 3):
            for c in range(C):
                fenetres.append([(r, c), (r + 1, c), (r + 2, c), (r + 3, c)])

        # diagonales (\)
        for r in range(R - 3):
            for c in range(C - 3):
                fenetres.append([(r, c), (r + 1, c + 1), (r + 2, c + 2), (r + 3, c + 3)])

        # diagonales (/)
        for r in range(3, R):
            for c in range(C - 3):
                fenetres.append([(r, c), (r - 1, c + 1), (r - 2, c + 2), (r - 3, c + 3)])

        # compter
        for fen in fenetres:
            nb_j = 0
            nb_adv = 0
            for (r, c) in fen:
                v = etat[r][c]
                if v == joueur:
                    nb_j += 1
                elif v == adversaire:
                    nb_adv += 1

            if nb_adv == 0:
                if nb_j == longueur:
                    total += 1

        return total
    
class Sylvain(AgentBase):
    nom = "Sylvain"

    def __init__(
        self,
        poids_centre: float = 3.0,
        poids_2: float = 1.0,
        poids_3: float = 5.0,
        poids_4: float = 1_000_000.0,
        poids_blocage_3: float = 6.0,
    ) -> None:
        super().__init__()

        self.poids_centre = poids_centre
        self.poids_2 = poids_2
        self.poids_3 = poids_3
        self.poids_4 = poids_4
        self.poids_blocage_3 = poids_blocage_3

        self.ordre_colonnes = [3, 2, 4, 1, 5, 0, 6]

    def choisir_action(self, etat, actions_valides, joueur):
        adversaire = -joueur

        meilleure_action = None
        meilleur_score = None
        base = grille_depuis_etat(etat)

        for action in actions_valides:
            plateau2 = GrillePuissance4(base)
            plateau2.appliquer_action(action, joueur)

            # Si ce coup gagne immédiatement, on joue sans hésiter
            fini, vainqueur = plateau2.etat_terminal()
            if fini and vainqueur == joueur:
                return action

            score = self._score_grille(plateau2.etat, joueur, adversaire)

            if meilleur_score is None:
                meilleur_score = score
                meilleure_action = action
            else:
                if score > meilleur_score:
                    meilleur_score = score
                    meilleure_action = action
                elif score == meilleur_score:
                    if self._est_preferable(action, meilleure_action):
                        meilleure_action = action

        if meilleure_action is None:
            return random.choice(actions_valides)

        return meilleure_action

    def _est_preferable(self, action: int, action_actuelle: int) -> bool:
        rang = self.ordre_colonnes.index(action)
        rang_actuel = self.ordre_colonnes.index(action_actuelle)
        return rang < rang_actuel

    def _score_grille(self, etat, joueur: int, adversaire: int) -> float:
        score = 0.0

        # (A) centre : compter les pions dans la colonne 3
        score += self.poids_centre * self._compter_pions_colonne(etat, colonne=3, joueur=joueur)

        # (B) fenêtres de 4 : on ajoute des points pour mes opportunités
        score += self._score_fenetres(etat, joueur=joueur, adversaire=adversaire)

        # (C) fenêtres de 4 adverses : on retire des points si l’adversaire a des opportunités
        score -= self._score_fenetres(etat, joueur=adversaire, adversaire=joueur, mode_adversaire=True)

        return score

    def _compter_pions_colonne(self, etat, colonne: int, joueur: int) -> int:
        nb = 0
        for r in range(GrillePuissance4.NB_LIGNES):
            if etat[r][colonne] == joueur:
                nb += 1
        return nb

    def _score_fenetres(self, etat, joueur: int, adversaire: int, mode_adversaire: bool = False) -> float:
        """
        Parcourt toutes les fenêtres de 4 cases.
        On ne score que les fenêtres qui ne contiennent PAS de pion adverse.
        - si 2 pions joueur -> +poids_2
        - si 3 pions joueur -> +poids_3
        - si 4 pions joueur -> +poids_4 (très grand)
        Si mode_adversaire=True, on renforce les menaces adverses de 3 (pour favoriser le blocage).
        """
        total = 0.0

        R = GrillePuissance4.NB_LIGNES
        C = GrillePuissance4.NB_COLONNES

        # horizontales
        for r in range(R):
            for c in range(C - 3):
                total += self._score_fenetre_unique(etat, joueur, adversaire, [(r, c), (r, c + 1), (r, c + 2), (r, c + 3)], mode_adversaire)

        # verticales
        for r in range(R - 3):
            for c in range(C):
                total += self._score_fenetre_unique(etat, joueur, adversaire, [(r, c), (r + 1, c), (r + 2, c), (r + 3, c)], mode_adversaire)

        # diagonales (\)
        for r in range(R - 3):
            for c in range(C - 3):
                total += self._score_fenetre_unique(etat, joueur, adversaire, [(r, c), (r + 1, c + 1), (r + 2, c + 2), (r + 3, c + 3)], mode_adversaire)

        # diagonales (/)
        for r in range(3, R):
            for c in range(C - 3):
                total += self._score_fenetre_unique(etat, joueur, adversaire, [(r, c), (r - 1, c + 1), (r - 2, c + 2), (r - 3, c + 3)], mode_adversaire)

        return total

    def _score_fenetre_unique(self, etat, joueur: int, adversaire: int, coords, mode_adversaire: bool) -> float:
        nb_j = 0
        nb_adv = 0
        nb_vides = 0

        for (r, c) in coords:
            v = etat[r][c]
            if v == joueur:
                nb_j += 1
            elif v == adversaire:
                nb_adv += 1
            else:
                nb_vides += 1

        # Si la fenêtre contient un pion adverse, elle n'est pas une opportunité "pure" pour joueur.
        if nb_adv > 0:
            return 0.0

        # Fenêtre exploitable : on score selon nb_j
        if nb_j == 4:
            return self.poids_4
        if nb_j == 3 and nb_vides == 1:
            # si c'est l'adversaire qu'on évalue, on met un poids un peu plus fort
            if mode_adversaire:
                return self.poids_blocage_3
            return self.poids_3
        if nb_j == 2 and nb_vides == 2:
            return self.poids_2

        return 0.0
