# ============================================================
# VaR avancee + Expected Shortfall + Stress Testing
# Auteur  : raphaelquant
# Date    : Mai 2026
# GitHub  : https://github.com/raphaelquant
# ============================================================
# Ce script compare les 3 methodes de VaR (historique,
# parametrique, Monte Carlo), calcule l'Expected Shortfall
# et realise des stress tests sur un portefeuille de 1M EUR.
#
# Donnees : CAC40 (^FCHI) — 2020 a 2024
# Reference : Bale III — adoption de l'ES comme mesure
#             de risque standard en remplacement de la VaR
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

# ============================================================
# 1. TELECHARGEMENT DES DONNEES — CAC40
# ============================================================

ticker = "^FCHI"
data = yf.download(
    ticker,
    start="2020-01-01",
    end="2024-12-31",
    auto_adjust=True
)

prix      = data["Close"]
rendements = prix.pct_change().dropna()
rendements = rendements.squeeze()

print("=" * 55)
print("   VAR AVANCEE + ES + STRESS TEST — CAC40")
print("=" * 55)
print(f"\nPeriode      : 2020-01-01 -> 2024-12-31")
print(f"Observations : {len(rendements)} jours de rendements")

# ============================================================
# 2. PARAMETRES ANNUALISES
# ============================================================

# Rendement et volatilite annualises pour Monte Carlo
# mu    = moyenne journaliere x 252
# sigma = ecart-type journalier x sqrt(252)
mu    = rendements.mean() * 252
sigma = rendements.std()  * 252**0.5

# ============================================================
# 3. VAR 95% — 3 METHODES
# ============================================================

# --- VaR Historique ---
# Percentile 5% des vrais rendements observes
# Avantage : capture les fat tails reels
# Limite   : depend du passe, pas predictif
VaR_hist  = np.percentile(rendements, 5)

# --- VaR Parametrique ---
# Suppose que les rendements suivent une loi normale
# z-score 95% = 1.645
# Avantage : rapide, analytique
# Limite   : sous-estime les evenements extremes
VaR_param = rendements.mean() - 1.645 * rendements.std()

# --- VaR Monte Carlo ---
# Simule 10 000 rendements journaliers via loi normale
# Parametres journaliers : mu/252 et sigma/sqrt(252)
# Avantage : flexible, extensible a tout produit
# Limite   : depend des hypotheses du modele
simulations = np.random.normal(mu/252, sigma/np.sqrt(252), 10000)
VaR_MC      = np.percentile(simulations, 5)

# ============================================================
# 4. EXPECTED SHORTFALL 95% — 3 METHODES
# ============================================================

# L'ES mesure la PERTE MOYENNE dans les 5% pires scenarios
# ES > VaR toujours — elle capture ce que la VaR ignore
# Bale III a adopte l'ES car elle est sous-additive (coherente)

# ES Historique : moyenne des rendements sous la VaR hist
ES_hist  = rendements[rendements < VaR_hist].mean()

# ES Monte Carlo : moyenne des simulations sous la VaR MC
ES_MC    = simulations[simulations < VaR_MC].mean()

# ES Parametrique : formule analytique via densite normale
# phi(z) / alpha avec z = 1.645 et alpha = 0.05
ES_param = rendements.mean() - (stats.norm.pdf(1.645) / 0.05) * rendements.std()

# ============================================================
# 5. STRESS TESTING
# ============================================================

# Simulation de 3 scenarios de crise sur un portefeuille 1M EUR
# Ces scenarios ne dependent d'aucune hypothese statistique
# Ils sont choisis par le risk manager selon son expertise
portefeuille     = 1_000_000
stress_krach     = portefeuille * (-0.30)  # Krach brutal type 2008
stress_taux      = portefeuille * (-0.15)  # Hausse brutale des taux
stress_liquidite = portefeuille * (-0.20)  # Crise de liquidite

# ============================================================
# 6. AFFICHAGE DES RESULTATS
# ============================================================

print("\n--- VaR 95% (journaliere) ---")
print(f"  VaR Historique   : {VaR_hist*100:.2f}%")
print(f"  VaR Parametrique : {VaR_param*100:.2f}%")
print(f"  VaR Monte Carlo  : {VaR_MC*100:.2f}%")

print("\n--- Expected Shortfall 95% ---")
print(f"  ES Historique    : {ES_hist*100:.2f}%")
print(f"  ES Parametrique  : {ES_param*100:.2f}%")
print(f"  ES Monte Carlo   : {ES_MC*100:.2f}%")

print("\n--- Verification ES > VaR (coherence Bale III) ---")
print(f"  |ES hist| > |VaR hist| : {abs(ES_hist) > abs(VaR_hist)}")
print(f"  |ES MC|   > |VaR MC|   : {abs(ES_MC)   > abs(VaR_MC)}")

print("\n--- Stress Tests (portefeuille 1 000 000 EUR) ---")
print(f"  Krach -30%            : {stress_krach:,.0f} EUR")
print(f"  Hausse taux -15%      : {stress_taux:,.0f} EUR")
print(f"  Crise liquidite -20%  : {stress_liquidite:,.0f} EUR")

# ============================================================
# 7. GRAPHIQUES
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "VaR avancee + Expected Shortfall + Stress Testing\n"
    "CAC40 (2020-2024)",
    fontsize=13
)

# --- Graphique 1 : Distribution des rendements + VaR ---
# Histogramme des rendements reels avec les 3 seuils VaR
# Permet de voir visuellement les differences entre methodes
axes[0].hist(
    rendements,
    bins=80,
    color="#2980b9",
    alpha=0.7,
    edgecolor="none",
    label="Rendements CAC40"
)
axes[0].axvline(
    x=VaR_hist, color="red",
    linewidth=2, linestyle="--",
    label=f"VaR Hist = {VaR_hist*100:.2f}%"
)
axes[0].axvline(
    x=VaR_param, color="orange",
    linewidth=2, linestyle="--",
    label=f"VaR Param = {VaR_param*100:.2f}%"
)
axes[0].axvline(
    x=VaR_MC, color="green",
    linewidth=2, linestyle="--",
    label=f"VaR MC = {VaR_MC*100:.2f}%"
)
axes[0].axvline(
    x=float(ES_hist), color="black",
    linewidth=2, linestyle=":",
    label=f"ES Hist = {float(ES_hist)*100:.2f}%"
)
axes[0].set_title("Distribution des rendements — VaR et ES")
axes[0].set_xlabel("Rendement journalier")
axes[0].set_ylabel("Frequence")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# --- Graphique 2 : Stress Tests ---
# Barres horizontales montrant les pertes en euros
# par scenario de crise
scenarios = ["Krach -30%", "Hausse taux -15%", "Crise liquidite -20%"]
pertes    = [stress_krach, stress_taux, stress_liquidite]
couleurs  = ["#e74c3c", "#e67e22", "#8e44ad"]

bars = axes[1].barh(scenarios, pertes, color=couleurs, alpha=0.8)
axes[1].axvline(x=0, color="black", linewidth=0.8)
axes[1].set_title("Stress Tests — Pertes sur portefeuille 1M EUR")
axes[1].set_xlabel("Perte (EUR)")
axes[1].grid(True, alpha=0.3, axis="x")

# Affiche les valeurs sur chaque barre
for bar, perte in zip(bars, pertes):
    axes[1].text(
        perte - 5000, bar.get_y() + bar.get_height()/2,
        f"{perte:,.0f} EUR",
        va="center", ha="right",
        color="white", fontsize=9, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("var_analysis.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegarde : var_analysis.png")
plt.show()