# ============================================================
# VaR paramétrique vs VaR historique — Expected Shortfall
# Ticker : CAC40 (^FCHI)
# Auteur : raphaelquant
# Date   : Avril 2026
# ============================================================

from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ============================================================
# 1. TELECHARGEMENT DES DONNEES
# ============================================================

# J'analyse le CAC40 comme proxy du marché français
# 5 ans de données pour avoir un historique robuste
# incluant le crash Covid 2020 et la hausse des taux 2022
ticker = "^FCHI"
data = yf.download(
    ticker,
    start="2020-01-01",
    end="2024-12-31",
    auto_adjust=True
)

prix = data["Close"]
rendements = prix.pct_change().dropna()

print("=" * 55)
print("   VaR PARAMETRIQUE vs HISTORIQUE — CAC40")
print("=" * 55)
print(f"\nPériode      : 2020-01-01 → 2024-12-31")
print(f"Observations : {len(rendements)} jours de rendements")

# ============================================================
# 2. PARAMETRES DE LA DISTRIBUTION
# ============================================================

# Rendement annuel = moyenne journalière × 252 jours de bourse
# La moyenne est proche de zéro sur les rendements journaliers
mu = rendements.mean() * 252

# Volatilité annuelle = écart-type journalier × √252
# La variance s'additionne linéairement, pas l'écart-type
sigma = rendements.std() * 252 ** 0.5

print(f"\n--- Paramètres annualisés ---")
print(f"Rendement moyen (mu)    : {float(mu)*100:.2f}%")
print(f"Volatilité (sigma)      : {float(sigma)*100:.2f}%")

# ============================================================
# 3. VAR PARAMETRIQUE
# ============================================================

# La VaR paramétrique suppose que les rendements suivent
# une loi normale — on utilise le z-score à 95% = 1.645
# VaR = mu - 1.645 × sigma
# Avantage : rapide et simple
# Limite : sous-estime les fat tails — dangereuse en crise
var_param = mu - 1.645 * sigma

# ============================================================
# 4. VAR HISTORIQUE
# ============================================================

# La VaR historique ne fait aucune hypothèse sur la distribution
# Elle lit directement le 5ème percentile des vraies données
# Plus réaliste car capture les fat tails observés
var_hist = np.percentile(rendements, 5)

# ============================================================
# 5. EXPECTED SHORTFALL
# ============================================================

# La VaR dit "tu ne perds pas plus de X dans 95% des cas"
# Mais que se passe-t-il dans les 5% pires cas ?
# L'ES mesure la perte MOYENNE au-delà de la VaR
# C'est pour ça que Bâle III utilise l'ES plutôt que la VaR
es = rendements[rendements < var_hist].mean()

# ============================================================
# 6. AFFICHAGE DES RESULTATS
# ============================================================

print(f"\n--- Value at Risk (annualisée) ---")
print(f"VaR paramétrique   : {float(var_param)*100:.2f}%")
print(f"VaR historique     : {float(var_hist)*100:.2f}%")
print(f"Expected Shortfall : {float(es)*100:.2f}%")
print(f"Différence hist vs param : {(float(var_hist)-float(var_param))*100:.2f}%")

# Si VaR hist > VaR param en valeur absolue
# → les vraies pertes extrêmes dépassent ce que la normale prédit
# → fat tails présents → loi normale insuffisante
if abs(float(var_hist)) > abs(float(var_param)):
    print("\nConclusion : Fat tails détectés")
    print("La loi normale SOUS-ESTIME le risque réel du CAC40")
else:
    print("\nConclusion : Distribution proche de la normale")

# ============================================================
# 7. GRAPHIQUES
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle(
    "VaR paramétrique vs historique — CAC40 (2020-2024)",
    fontsize=14
)

# --- Graphique 1 : Distribution des rendements ---
# Je veux voir visuellement si les queues sont plus épaisses
# que ce que la loi normale prédirait — les fat tails
axes[0].hist(
    rendements,
    bins=60,
    color="#2980b9",
    alpha=0.7,
    edgecolor="none",
    label="Rendements observés"
)
axes[0].axvline(
    x=float(var_hist),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"VaR hist = {float(var_hist)*100:.2f}%"
)
axes[0].axvline(
    x=float(var_param),
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"VaR param = {float(var_param)*100:.2f}%"
)
axes[0].axvline(
    x=float(es),
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"ES = {float(es)*100:.2f}%"
)
axes[0].set_title("Distribution des rendements journaliers — CAC40")
axes[0].set_xlabel("Rendement journalier")
axes[0].set_ylabel("Fréquence")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Graphique 2 : Rendements journaliers dans le temps ---
# Barres vertes = jours positifs / rouges = jours négatifs
# La ligne rouge montre le seuil de la VaR historique
couleurs = rendements.squeeze().apply(
    lambda x: "#2ecc71" if x >= 0 else "#e74c3c"
)
axes[1].bar(
    rendements.index,
    rendements.squeeze(),
    color=couleurs,
    width=1,
    alpha=0.8
)
axes[1].axhline(
    y=float(var_hist),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"VaR hist = {float(var_hist)*100:.2f}%"
)
axes[1].set_title("Rendements journaliers — CAC40")
axes[1].set_ylabel("Rendement journalier")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("var_hist_vs_param.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegardé : var_hist_vs_param.png")
plt.show()