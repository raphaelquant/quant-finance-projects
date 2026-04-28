# ============================================================
# Analyse 3 actifs — CAC40, TotalEnergies, LVMH
# Auteur : raphaelquant
# Date   : Avril 2026
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tickers choisis pour l'analyse
# CAC40 comme indice de référence
# TotalEnergies et LVMH comme actions françaises majeures
tickers = ["^FCHI", "TTE.PA", "MC.PA"]

# Téléchargement des données sur 5 ans
# J'ai choisi 2020 pour inclure le crash Covid et la reprise
data = yf.download(
    tickers,
    start="2020-01-01",
    end="2024-12-31",
    auto_adjust=True
)

prix = data["Close"]

print("=" * 55)
print("   ANALYSE 3 ACTIFS — CAC40 | TTE.PA | MC.PA")
print("=" * 55)
print(f"\nPériode     : 2020-01-01 → 2024-12-31")
print(f"Nb de jours : {len(prix)}")

# Calcul des rendements journaliers
# pct_change() donne le rendement entre chaque jour
rendements = prix.pct_change().dropna()

print(f"Observations : {len(rendements)} rendements journaliers")

# Matrice de corrélation entre les 3 actifs
# Je veux voir si TTE et MC suivent vraiment le CAC40
# ou s'ils ont des dynamiques propres
correlation = rendements.corr()

print("\n--- Matrice de corrélation ---")
print(correlation.round(2))

# Pondération du portefeuille
# J'ai choisi de surpondérer le CAC40 pour avoir
# une base stable, et répartir le reste entre TTE et MC
# Contrainte : les poids doivent sommer à 1
poids = np.array([0.40, 0.30, 0.30])

print(f"\n--- Pondération choisie ---")
for i, ticker in enumerate(tickers):
    print(f"{ticker} : {poids[i]*100:.0f}%")
print(f"Total      : {sum(poids)*100:.0f}%")

# Rendement journalier du portefeuille
# dot() fait la somme pondérée pour chaque jour automatiquement
rendement_port = rendements.dot(poids)

# Rendements cumulés
# J'utilise cumprod() et pas cumsum() parce que les rendements
# se composent dans le temps — chaque gain est réinvesti
# (1+r1) x (1+r2) x ... est la bonne formule, pas r1+r2+...
cumules = (1 + rendements).cumprod()
cumul_port = (1 + rendement_port).cumprod()

# Performance totale sur la période
print("\n--- Performance totale 2020-2024 ---")
for ticker in tickers:
    gain = (cumules[ticker].iloc[-1] - 1) * 100
    print(f"{ticker} : {gain:.2f}%")
gain_port = (cumul_port.iloc[-1] - 1) * 100
print(f"Portefeuille : {gain_port:.2f}%")

# Graphiques
fig, axes = plt.subplots(3, 1, figsize=(12, 15))
fig.suptitle(
    "Analyse 3 actifs — CAC40 | TotalEnergies | LVMH\n2020-2024",
    fontsize=14
)

# Graphique 1 — Evolution comparée des 3 actifs
# Base 1€ pour comparer les performances sur la même échelle
couleurs = ["#2980b9", "#e67e22", "#8e44ad"]
for i, ticker in enumerate(tickers):
    axes[0].plot(
        cumules[ticker],
        label=ticker,
        linewidth=1.8,
        color=couleurs[i]
    )
axes[0].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
axes[0].set_title("Evolution des 3 actifs (base 1€ en janvier 2020)")
axes[0].set_ylabel("Valeur du portefeuille")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Graphique 2 — Performance du portefeuille combiné
# Je veux voir si la diversification améliore le profil risque/rendement
axes[1].plot(
    cumul_port,
    label=f"Portefeuille 40% CAC40 / 30% TTE / 30% LVMH",
    color="black",
    linewidth=2
)
axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)
axes[1].set_title("Performance du portefeuille pondéré")
axes[1].set_ylabel("Valeur du portefeuille")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Graphique 3 — Heatmap de corrélation
# Rouge = corrélation positive forte (bougent ensemble)
# Bleu  = corrélation négative (bougent en sens inverse)
im = axes[2].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
axes[2].set_xticks([0, 1, 2])
axes[2].set_xticklabels(tickers)
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(tickers)
axes[2].set_title("Matrice de corrélation entre les 3 actifs")

# Valeurs affichées dans chaque case
for i in range(3):
    for j in range(3):
        axes[2].text(
            j, i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center", va="center",
            color="black", fontsize=12
        )

plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig("analyse_3_actifs.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegardé : analyse_3_actifs.png")
plt.show()