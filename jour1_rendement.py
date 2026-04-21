# Auteur : raphaelquant | Date : Avril 2026
# Description : calcul des rendements journaliers d'une action

import numpy as np

prix = [100, 102, 98, 105, 103, 108, 106, 110, 107, 112]

rendements = [(prix[i] - prix[i-1]) / prix[i-1] * 100
              for i in range(1, len(prix))]

print("=== ANALYSE DES RENDEMENTS ===")
print(f"Rendement moyen : {np.mean(rendements):.4f}%")
print(f"Volatilité : {np.std(rendements):.4f}%")
print(f"Rendement max : {max(rendements):.4f}%")
print(f"Rendement min : {min(rendements):.4f}%")
print(f"Rendement cumulé : {(prix[-1]/prix[0]-1)*100:.2f}%")