# ============================================================
# Données Réelles avec yfinance
# Tickers : TTE.PA (TotalEnergies), OR.PA (L'Oréal), ^FCHI (CAC40)
# Concepts : DataFrame, rendements journaliers, annualisation ×252
# Auteur   : raphaelquant
# ============================================================

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. TELECHARGEMENT DES DONNEES
# ============================================================

# Tickers Yahoo Finance
# TTE.PA = TotalEnergies coté à Paris
# OR.PA  = L'Oréal coté à Paris
# ^FCHI  = Indice CAC 40
tickers = ["TTE.PA", "OR.PA", "^FCHI"]

# Téléchargement depuis Yahoo Finance
# auto_adjust=True intègre les dividendes dans les prix
data = yf.download(
    tickers,
    start="2023-01-01",
    end="2024-12-31",
    auto_adjust=True
)

# Extraction des prix de clôture uniquement
# data["Close"] retourne un DataFrame avec une colonne par ticker
prix = data["Close"]

print("=" * 55)
print("   DONNEES REELLES — yfinance")
print("=" * 55)
print(f"\nPériode     : 2023-01-01 → 2024-12-31")
print(f"Nb de jours : {len(prix)}")
print("\n--- Aperçu prix de clôture (5 premiers jours) ---")
print(prix.head())

# ============================================================
# 2. CALCUL DES RENDEMENTS JOURNALIERS
# ============================================================

# pct_change() = (Prix_j - Prix_j-1) / Prix_j-1
# Chaque ligne = rendement de ce jour par rapport au jour précédent
# dropna() supprime la première ligne qui est NaN (pas de j-1)
rendements = prix.pct_change().dropna()

print("\n--- Aperçu rendements journaliers (5 premiers jours) ---")
print(rendements.head())

# ============================================================
# 3. ANNUALISATION
# ============================================================

# Il y a 252 jours de bourse par an (pas 365 — marchés fermés weekends)
# Rendement annuel    = moyenne journalière × 252
# Volatilité annuelle = écart-type journalier × √252
# Pourquoi √252 ? La variance s'additionne linéairement,
# donc l'écart-type s'additionne en racine carrée
JOURS = 252

rendement_annuel   = rendements.mean() * JOURS
volatilite_annuelle = rendements.std() * (JOURS ** 0.5)

# Ratio de Sharpe avec taux sans risque = 2%
rf = 0.02
sharpe = (rendement_annuel - rf) / volatilite_annuelle

print("\n--- Résultats annualisés ---")
print(f"{'Ticker':<10} {'Rendement':>12} {'Volatilité':>12} {'Sharpe':>10}")
print("-" * 48)
for ticker in tickers:
    r = rendement_annuel[ticker] * 100
    v = volatilite_annuelle[ticker] * 100
    s = sharpe[ticker]
    print(f"{ticker:<10} {r:>11.2f}% {v:>11.2f}% {s:>10.4f}")

# ============================================================
# 4. GRAPHIQUES
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle(
    "Analyse — TTE.PA | OR.PA | CAC40\n2023-2024",
    fontsize=15,
    fontweight='bold'
)

# --- Graphique 1 : Prix normalisés base 100 ---
# Permet de comparer l'évolution relative des 3 actifs
# sur la même échelle (comme le graphique Apple/S&P500)
prix_normalise = (prix / prix.iloc[0]) * 100

axes[0].set_title("Prix normalisés (base 100 au 01/01/2023)")
couleurs = ["#2980b9", "#e67e22", "#8e44ad"]
for i, ticker in enumerate(tickers):
    axes[0].plot(
        prix_normalise[ticker],
        label=ticker,
        linewidth=1.8,
        color=couleurs[i]
    )
axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Base 100')
axes[0].legend()
axes[0].set_ylabel("Prix normalisé")
axes[0].grid(True, alpha=0.3)

# --- Graphique 2 : Rendements journaliers TTE.PA ---
# Les barres vertes = jours positifs / rouges = jours négatifs
axes[1].set_title("Rendements journaliers — TotalEnergies (TTE.PA)")
couleurs_barres = rendements["TTE.PA"].apply(
    lambda x: "#2ecc71" if x >= 0 else "#e74c3c"
)
axes[1].bar(
    rendements.index,
    rendements["TTE.PA"],
    color=couleurs_barres,
    width=1,
    alpha=0.8
)
axes[1].axhline(y=0, color='black', linewidth=0.8)
axes[1].set_ylabel("Rendement journalier")
axes[1].grid(True, alpha=0.3)

# --- Graphique 3 : Distribution des rendements ---
# Montre si les rendements suivent une loi normale
# (rappel : fat tails en pratique)
axes[2].set_title("Distribution des rendements journaliers (histogramme)")
for i, ticker in enumerate(tickers):
    axes[2].hist(
        rendements[ticker],
        bins=60,
        alpha=0.5,
        label=ticker,
        color=couleurs[i],
        edgecolor='none'
    )
axes[2].axvline(x=0, color='black', linewidth=1, linestyle='--')
axes[2].set_xlabel("Rendement journalier")
axes[2].set_ylabel("Fréquence")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Sauvegarde PNG
plt.savefig("donnees_reelles.png", dpi=150, bbox_inches='tight')
print("\nGraphique sauvegardé : donnees_reelles.png ✓")
plt.show()