# ============================================================
# Matrice de covariance — TotalEnergies, LVMH, L'Oréal
# Auteur : raphaelquant
# Date   : Avril 2026
# ============================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ============================================================
# 1. TELECHARGEMENT DES DONNEES
# ============================================================

# J'analyse 3 grandes capitalisations françaises
# TTE.PA = TotalEnergies — secteur énergie
# MC.PA  = LVMH — secteur luxe
# OR.PA  = L'Oréal — secteur cosmétique
# 5 ans de données pour une covariance robuste
tickers = ["TTE.PA", "MC.PA", "OR.PA"]

data = yf.download(
    tickers,
    start="2020-01-01",
    end="2024-12-31",
    auto_adjust=True
)

prix = data["Close"]

print("=" * 55)
print("   MATRICE DE COVARIANCE — TTE | MC | OR")
print("=" * 55)
print(f"\nPériode      : 2020-01-01 → 2024-12-31")
print(f"Observations : {len(prix)} jours")

# ============================================================
# 2. RENDEMENTS ET MATRICE DE COVARIANCE
# ============================================================

# Rendements journaliers
rendements = prix.pct_change().dropna()

# Matrice de covariance annualisée
# .cov() calcule la covariance journalière entre chaque paire
# × 252 pour annualiser — les covariances s'additionnent
# comme les variances donc on multiplie directement par 252
cov_matrix = rendements.cov() * 252

print("\n--- Matrice de covariance annualisée ---")
print(cov_matrix.round(4))

# ============================================================
# 3. PORTEFEUILLE EQUIPONDÉRÉ
# ============================================================

# Poids égaux — 1/3 pour chaque actif
# C'est le portefeuille le plus simple et souvent difficile à battre
# Contrainte : w1 + w2 + w3 = 1
w = np.array([1/3, 1/3, 1/3])

# Variance du portefeuille via la formule matricielle
# σ²p = w^T × Σ × w
# L'opérateur @ fait le produit matriciel en Python
# Dimensions : (1×3) × (3×3) × (3×1) = scalaire
variance_port = w @ cov_matrix @ w

# Volatilité = racine carrée de la variance
# C'est la mesure de risque qu'on compare au Sharpe
vol_port = np.sqrt(variance_port)

print(f"\n--- Portefeuille équipondéré (1/3 chaque) ---")
print(f"Variance du portefeuille  : {variance_port*100:.2f}%")
print(f"Volatilité du portefeuille: {vol_port*100:.2f}%")

# ============================================================
# 4. HEATMAP DE LA MATRICE DE COVARIANCE
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fig.suptitle(
    "Matrice de covariance annualisée\nTotalEnergies | LVMH | L'Oréal",
    fontsize=13
)

# Heatmap — rouge = covariance forte / bleu = faible
# Je veux voir quels actifs bougent le plus ensemble
# pour identifier les risques de concentration
im = ax.imshow(cov_matrix, cmap="coolwarm")

# Labels des axes
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(tickers)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(tickers)
ax.set_title("Matrice de covariance annualisée")

# Valeurs affichées dans chaque case
# 4 décimales car les covariances sont petites en décimales
for i in range(3):
    for j in range(3):
        ax.text(
            j, i,
            f"{cov_matrix.iloc[i,j]:.4f}",
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

# Barre de couleur pour la légende
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("matrice_covariance.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegardé : matrice_covariance.png")
plt.show()