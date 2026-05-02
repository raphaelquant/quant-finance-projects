# ============================================================
# Monte Carlo — Simulation GBM 10 000 trajectoires
# Auteur  : raphaelquant
# Date    : Avril 2026
# GitHub  : https://github.com/raphaelquant
# ============================================================
# Ce script simule 10 000 trajectoires de prix via le
# Mouvement Brownien Geometrique (GBM) et calcule la VaR
# Monte Carlo par simulation stochastique.
#
# Formule GBM :
# S(t+dt) = S(t) x exp[(mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z]
# Z ~ N(0,1)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. PARAMETRES DU MODELE
# ============================================================

S0           = 100     # Prix initial de l'actif
mu           = 0.08    # Drift annuel — rendement moyen attendu (8%)
sigma        = 0.20    # Volatilite annualisee (20%)
T            = 1       # Horizon de simulation : 1 an
N            = 252     # Nombre de jours de bourse par an
n_simulations= 10000   # Nombre de trajectoires simulees
dt           = T / N   # Pas de temps journalier

# ============================================================
# 2. SIMULATION MONTE CARLO — MOUVEMENT BROWNIEN GEOMETRIQUE
# ============================================================

# Chocs aleatoires Z ~ N(0,1) — matrice (N jours x n_simulations)
# Chaque colonne = une trajectoire de prix independante
Z = np.random.normal(0, 1, (N, n_simulations))

# Rendements journaliers selon la formule GBM
# (mu - 0.5*sigma^2)*dt = composante deterministe (drift corrige)
# sigma*sqrt(dt)*Z       = composante aleatoire (choc de marche)
rendements = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z

# Trajectoires de prix via produit cumulatif
# np.cumprod multiplie les facteurs exp(rendement) jour apres jour
# axis=0 = on accumule dans le sens du temps (lignes)
prix = S0 * np.cumprod(np.exp(rendements), axis=0)

# ============================================================
# 3. CALCUL DE LA VAR MONTE CARLO
# ============================================================

# Prix finaux = derniere ligne de la matrice = prix apres 1 an
prix_finaux = prix[-1, :]

# VaR 95% = percentile 5% des prix finaux
# Dans 95% des simulations le prix ne descend pas en dessous de VaR_MC
VaR_MC    = np.percentile(prix_finaux, 5)
perte_pct = (VaR_MC - S0) / S0 * 100

print("=" * 55)
print("   MONTE CARLO — SIMULATION GBM 10 000 TRAJECTOIRES")
print("=" * 55)

print("\n--- Parametres ---")
print(f"  Prix initial (S0)      : {S0} EUR")
print(f"  Drift annuel (mu)      : {mu*100:.1f}%")
print(f"  Volatilite (sigma)     : {sigma*100:.1f}%")
print(f"  Horizon                : {T} an ({N} jours)")
print(f"  Simulations            : {n_simulations:,}")

print("\n--- Resultats ---")
print(f"  Prix moyen simule      : {np.mean(prix_finaux):.2f} EUR")
print(f"  Prix median simule     : {np.median(prix_finaux):.2f} EUR")
print(f"  Prix min simule        : {np.min(prix_finaux):.2f} EUR")
print(f"  Prix max simule        : {np.max(prix_finaux):.2f} EUR")

print("\n--- VaR Monte Carlo 95% ---")
print(f"  VaR MC (prix plancher) : {VaR_MC:.2f} EUR")
print(f"  Perte maximale 95%     : {perte_pct:.2f}%")
print(f"  Interpretation         : dans 95% des simulations,")
print(f"                           le prix reste au-dessus de {VaR_MC:.2f} EUR")

# ============================================================
# 4. GRAPHIQUES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"Monte Carlo GBM — {n_simulations:,} simulations\n"
    f"S0={S0} | mu={mu*100:.0f}% | sigma={sigma*100:.0f}% | T={T}an",
    fontsize=13
)

# --- Graphique 1 : Trajectoires de prix ---
# On affiche seulement 200 trajectoires pour la lisibilite
axes[0].plot(prix[:, :5000], alpha=0.05, color="#185FA5", linewidth=0.5)
axes[0].axhline(y=S0, color="black", linewidth=1.2,
                linestyle="--", label=f"Prix initial S0={S0}")
axes[0].axhline(y=VaR_MC, color="red", linewidth=1.5,
                linestyle="--", label=f"VaR 95% = {VaR_MC:.1f} EUR")
axes[0].set_title("Trajectoires de prix simulees (200/10 000)")
axes[0].set_xlabel("Jours")
axes[0].set_ylabel("Prix (EUR)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Graphique 2 : Distribution des prix finaux ---
# Histogramme des 10 000 prix finaux apres 1 an
axes[1].hist(prix_finaux, bins=100, color="#185FA5",
             alpha=0.7, edgecolor="none", label="Prix finaux")
axes[1].axvline(x=VaR_MC, color="red", linewidth=2,
                linestyle="--", label=f"VaR 95% = {VaR_MC:.1f} EUR")
axes[1].axvline(x=S0, color="black", linewidth=1.5,
                linestyle="--", label=f"Prix initial = {S0} EUR")
axes[1].set_title("Distribution des prix finaux apres 1 an")
axes[1].set_xlabel("Prix final (EUR)")
axes[1].set_ylabel("Frequence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("monte_carlo.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegarde : monte_carlo.png")
plt.show()