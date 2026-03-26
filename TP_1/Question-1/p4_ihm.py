import tkinter as tk
from tkinter import ttk

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



CELL_SIZE = 80
PADDING = 20
BOARD_BG = "#1c4fd7"
EMPTY_COLOR = "#f4f4f4"
P1_COLOR = "#e53935"  # humain
P2_COLOR = "#fdd835"  # IA


class Puissance4IHM:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Puissance 4 - IHM")

        self.env = GrillePuissance4()
        self.env.reinitialiser()
        self.partie_terminee = False
        self.vainqueur = None

        self.joueur_humain = +1
        self.joueur_ia = -1
        self.joueur_courant = +1

        self.agents = {
            "JeanClaude": JeanClaude(self.env),
            "Nicolas": Nicolas(self.env),
            "Isabelle": Isabelle(self.env),
            "Sylvie": Sylvie(self.env),
            "Gilles": Gilles(),
            "Nathalie": Nathalie(),
            "Emmanuelle": Emmanuelle(self.env),
            "Pierre": Pierre(),
            "Paul": Paul(),
            "Sylvain": Sylvain(self.env),
        }

        self.agent_ia_nom = tk.StringVar(value="JeanClaude")
        self.ia_commence = tk.BooleanVar(value=False)

        self._build_ui()
        self._dessiner_plateau()

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=12)

        ttk.Label(controls, text="IA:").pack(side="left")
        ia_menu = ttk.OptionMenu(
            controls,
            self.agent_ia_nom,
            self.agent_ia_nom.get(),
            *self.agents.keys(),
        )
        ia_menu.pack(side="left", padx=(6, 18))

        ttk.Checkbutton(
            controls,
            text="L'IA commence",
            variable=self.ia_commence,
        ).pack(side="left", padx=(0, 18))

        ttk.Button(controls, text="Nouvelle partie", command=self.nouvelle_partie).pack(
            side="left"
        )

        self.status = ttk.Label(self.root, text="À vous de jouer.")
        self.status.pack(fill="x", padx=12, pady=(0, 8))

        width = self.env.NB_COLONNES * CELL_SIZE + 2 * PADDING
        height = self.env.NB_LIGNES * CELL_SIZE + 2 * PADDING
        self.canvas = tk.Canvas(
            self.root, width=width, height=height, bg=BOARD_BG, highlightthickness=0
        )
        self.canvas.pack(padx=12, pady=(0, 12))
        self.canvas.bind("<Button-1>", self._on_click)

    def _dessiner_plateau(self) -> None:
        self.canvas.delete("all")
        for r in range(self.env.NB_LIGNES):
            for c in range(self.env.NB_COLONNES):
                x1 = PADDING + c * CELL_SIZE + 6
                y1 = PADDING + r * CELL_SIZE + 6
                x2 = x1 + CELL_SIZE - 12
                y2 = y1 + CELL_SIZE - 12

                v = self.env.etat[r][c]
                if v == self.joueur_humain:
                    color = P1_COLOR
                elif v == self.joueur_ia:
                    color = P2_COLOR
                else:
                    color = EMPTY_COLOR

                self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline="")

    def nouvelle_partie(self) -> None:
        self.env.reinitialiser()
        self.partie_terminee = False
        self.vainqueur = None
        self.joueur_courant = self.joueur_humain
        self._dessiner_plateau()

        if self.ia_commence.get():
            self.joueur_courant = self.joueur_ia
            self.status.config(text="L'IA commence...")
            self.root.after(300, self._tour_ia)
        else:
            self.status.config(text="À vous de jouer.")

    def _colonne_depuis_x(self, x: int) -> int:
        col = (x - PADDING) // CELL_SIZE
        if 0 <= col < self.env.NB_COLONNES:
            return int(col)
        return -1

    def _on_click(self, event) -> None:
        if self.partie_terminee or self.joueur_courant != self.joueur_humain:
            return

        col = self._colonne_depuis_x(event.x)
        if col < 0:
            return

        actions = self.env.actions_valides()
        if col not in actions:
            self.status.config(text="Colonne pleine. Essayez une autre colonne.")
            return

        self._jouer_coup(col, self.joueur_humain)

        if not self.partie_terminee:
            self.joueur_courant = self.joueur_ia
            self.status.config(text="L'IA réfléchit...")
            self.root.after(250, self._tour_ia)

    def _tour_ia(self) -> None:
        if self.partie_terminee or self.joueur_courant != self.joueur_ia:
            return

        agent = self.agents[self.agent_ia_nom.get()]
        etat_pour_agent = self.env.etat
        actions = self.env.actions_valides()
        action = agent.choisir_action(etat_pour_agent, actions, self.joueur_ia)
        self._jouer_coup(action, self.joueur_ia)

        if not self.partie_terminee:
            self.joueur_courant = self.joueur_humain
            self.status.config(text="À vous de jouer.")

    def _jouer_coup(self, action: int, joueur: int) -> None:
        self.env.appliquer_action(action, joueur)
        self._dessiner_plateau()

        self.partie_terminee, self.vainqueur = self.env.etat_terminal()
        if self.partie_terminee:
            if self.vainqueur == 0:
                self.status.config(text="Match nul.")
            elif self.vainqueur == self.joueur_humain:
                self.status.config(text="Vous gagnez !")
            else:
                self.status.config(text="L'IA gagne.")


if __name__ == "__main__":
    root = tk.Tk()
    app = Puissance4IHM(root)
    root.mainloop()
