# ============================================================
# Simulateur — Sharpe + VaR 95% sur données réelles
# Ticker : TTE.PA (TotalEnergies)
# Concepts : VaR historique, Sharpe, annualisation ×252
# Auteur   : raphaelquant
# ============================================================

import yfinance as yf       # Télécharger données boursières
import numpy as np          # Calcul numérique (percentile pour VaR)
import pandas as pd         # Manipulation des données
import matplotlib.pyplot as plt  # Graphiques

# ============================================================
# 1. TELECHARGEMENT DES DONNEES
# ============================================================

ticker = "TTE.PA"           # TotalEnergies coté à Paris

# Téléchargement depuis Yahoo Finance
data = yf.download(
    ticker,
    start="2023-01-01",
    end="2024-12-31",
    auto_adjust=True        # Prix ajustés dividendes inclus
)

# Extraction des prix de clôture
r = data["Close"]

print("=" * 55)
print("   SIMULATEUR — SHARPE + VaR 95% — TTE.PA")
print("=" * 55)
print(f"\nNombre de jours : {len(r)}")

# ============================================================
# 2. CALCUL DES RENDEMENTS JOURNALIERS
# ============================================================

# pct_change() = (Prix_j - Prix_j-1) / Prix_j-1
# dropna() supprime le premier NaN
# On écrase r — maintenant r contient les rendements
r = r.pct_change().dropna()

print(f"Rendements journaliers calculés : {len(r)} observations")

# ============================================================
# 3. METRIQUES DE PERFORMANCE
# ============================================================

# Rendement annuel = moyenne journalière × 252
mu = r.mean() * 252

# Volatilité annuelle = écart-type journalier × √252
# Rappel : la variance s'additionne, pas l'écart-type
sigma = r.std() * 252 ** 0.5

# Ratio de Sharpe = rendement excédentaire / volatilité
# rf = 2% = taux sans risque
sharpe = (mu - 0.02) / sigma

# ============================================================
# 4. VAR 95% HISTORIQUE
# ============================================================

# np.percentile(r, 5) = 5ème percentile des rendements
# = la valeur en dessous de laquelle se trouvent les 5% pires journées
# = perte maximale dans 95% des cas sur 1 journée
# C'est une valeur NEGATIVE — c'est une perte
var_95 = np.percentile(r, 5)

# ============================================================
# 5. AFFICHAGE DES RESULTATS
# ============================================================

print("\n--- Métriques annualisées ---")
print(f"Rendement annuel    : {float(mu)*100:.2f}%")
print(f"Volatilité annuelle : {float(sigma)*100:.2f}%")
print(f"Ratio de Sharpe     : {float(sharpe):.4f}")
print("\n--- Value at Risk ---")
print(f"VaR 95% perte max 1 jour : {float(var_95)*100:.2f}%")

print(f"Interprétation : dans 95% des journées,")
print(f"TotalEnergies ne perd pas plus de {abs(var_95*100):.2f}%")

# ============================================================
# 6. GRAPHIQUES
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("TotalEnergies (TTE.PA) — Sharpe + VaR 95%", fontsize=14)

# --- Graphique 1 : Rendements journaliers ---
# Barres vertes = jours positifs / rouges = jours négatifs
couleurs = r.squeeze().apply(lambda x: "#2ecc71" if x >= 0 else "#e74c3c")
axes[0].bar(r.index, r.squeeze(), color=couleurs, width=1, alpha=0.8)
axes[0].axhline(y=0, color="black", linewidth=0.8)
axes[0].axhline(
    y=var_95,
    color="red",
    linewidth=2,
    linestyle="--",
    label=f"VaR 95% = {var_95*100:.2f}%"
)
axes[0].set_title("Rendements journaliers")
axes[0].set_ylabel("Rendement")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Graphique 2 : Distribution des rendements ---
# Histogramme pour visualiser les fat tails
axes[1].hist(r, bins=60, color="#2980b9", alpha=0.7, edgecolor="none")
axes[1].axvline(
    x=var_95,
    color="red",
    linewidth=2,
    linestyle="--",
    label=f"VaR 95% = {var_95*100:.2f}%"
)
axes[1].set_title("Distribution des rendements journaliers")
axes[1].set_xlabel("Rendement journalier")
axes[1].set_ylabel("Fréquence")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Sauvegarde PNG
plt.savefig("var_sharpe_TTE.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegardé : var_sharpe_TTE.png")
plt.show()