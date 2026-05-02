# ============================================================
# Frontiere Efficiente de Markowitz
# Auteur  : raphaelquant
# Date    : Mai 2026
# GitHub  : https://github.com/raphaelquant
# ============================================================
# Ce script construit la frontiere efficiente de Markowitz
# en simulant 5000 portefeuilles aleatoires via la distribution
# de Dirichlet, puis optimise le portefeuille tangent via
# scipy.optimize.minimize (maximisation du Sharpe ratio).
#
# Actifs : TTE.PA | MC.PA | OR.PA | ^FCHI (2020-2024)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# ============================================================
# 1. TELECHARGEMENT DES DONNEES
# ============================================================

tickers = ["TTE.PA", "MC.PA", "OR.PA", "^FCHI"]

data = yf.download(
    tickers,
    start="2020-01-01",
    end="2024-12-31",
    auto_adjust=True
)

prix      = data["Close"]
rendements = prix.pct_change().dropna()

print("=" * 55)
print("   FRONTIERE EFFICIENTE — MARKOWITZ")
print("=" * 55)
print(f"\nPeriode      : 2020-01-01 -> 2024-12-31")
print(f"Observations : {len(rendements)} jours")
print(f"Actifs       : {tickers}")

# ============================================================
# 2. PARAMETRES ANNUALISES
# ============================================================

# Rendements esperes annualises de chaque actif (x252)
# Matrice de covariance annualisee (x252)
mu_actifs  = rendements.mean() * 252
cov_matrix = rendements.cov()  * 252
n          = len(tickers)

# ============================================================
# 3. SIMULATION DE 5000 PORTEFEUILLES ALEATOIRES
# ============================================================

# np.random.dirichlet(np.ones(n)) genere des poids aleatoires
# qui somment exactement a 1 — distribution uniforme sur le
# simplex (triangle des portefeuilles valides)
rendements_port  = []
volatilites_port = []
sharpes_port     = []

for _ in range(5000):
    # Poids aleatoires via distribution de Dirichlet
    w = np.random.dirichlet(np.ones(n))

    # Rendement du portefeuille : produit scalaire poids x rendements
    rp = w @ mu_actifs

    # Volatilite via formule matricielle : sqrt(w^T x Sigma x w)
    vp = np.sqrt(w @ cov_matrix @ w)

    # Sharpe ratio : rendement excedentaire / volatilite
    sharpe = (rp - 0.02) / vp

    rendements_port.append(rp)
    volatilites_port.append(vp)
    sharpes_port.append(sharpe)

# Conversion en arrays numpy pour les graphiques
rendements_port  = np.array(rendements_port)
volatilites_port = np.array(volatilites_port)
sharpes_port     = np.array(sharpes_port)

# ============================================================
# 4. OPTIMISATION — PORTEFEUILLE TANGENT (SHARPE MAX)
# ============================================================

# On minimise le negatif du Sharpe = on maximise le Sharpe
# Contrainte : somme des poids = 1
# Bornes     : chaque poids entre 0 et 1 (pas de vente a decouvert)
def neg_sharpe(w):
    rp = w @ mu_actifs
    vp = np.sqrt(w @ cov_matrix @ w)
    return -(rp - 0.02) / vp

contraintes = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bornes      = [(0, 1)] * n
w0          = np.ones(n) / n  # poids initiaux : portefeuille equipondere

result    = minimize(neg_sharpe, w0, method="SLSQP",
                     bounds=bornes, constraints=contraintes)
w_optimal = result.x

# Metriques du portefeuille optimal
rp_opt     = w_optimal @ mu_actifs
vp_opt     = np.sqrt(w_optimal @ cov_matrix @ w_optimal)
sharpe_opt = (rp_opt - 0.02) / vp_opt

# ============================================================
# 5. AFFICHAGE DES RESULTATS
# ============================================================

print("\n--- Portefeuille tangent optimal ---")
print("  Allocation optimale :")
for i, ticker in enumerate(tickers):
    print(f"    {ticker:8s} : {w_optimal[i]*100:.2f}%")

print(f"\n  Rendement annuel  : {float(rp_opt)*100:.2f}%")
print(f"  Volatilite annuelle: {float(vp_opt)*100:.2f}%")
print(f"  Ratio de Sharpe   : {float(sharpe_opt):.4f}")

# ============================================================
# 6. GRAPHIQUE — FRONTIERE EFFICIENTE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle(
    "Frontiere efficiente de Markowitz\n"
    "TTE.PA | MC.PA | OR.PA | CAC40 — 2020-2024",
    fontsize=13
)

# Nuage de 5000 portefeuilles colores par Sharpe ratio
# Jaune = Sharpe eleve (efficient) / Bleu = Sharpe faible
sc = ax.scatter(
    volatilites_port,
    rendements_port,
    c=sharpes_port,
    cmap="viridis",
    alpha=0.5,
    s=10,
    label="Portefeuilles simules"
)
plt.colorbar(sc, ax=ax, label="Ratio de Sharpe")

# Portefeuille tangent — etoile rouge
ax.scatter(
    float(vp_opt), float(rp_opt),
    color="red", s=300,
    zorder=5, marker="*",
    label=f"Portefeuille optimal (Sharpe={float(sharpe_opt):.2f})"
)

ax.set_xlabel("Volatilite annualisee (risque)")
ax.set_ylabel("Rendement annualise")
ax.set_title("5000 portefeuilles simules via distribution de Dirichlet")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("markowitz.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegarde : markowitz.png")
plt.show()