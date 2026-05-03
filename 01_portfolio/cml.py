# ============================================================
# Capital Market Line (CML) — Markowitz + CML
# Auteur  : raphaelquant
# Date    : Mai 2026
# GitHub  : https://github.com/raphaelquant
# ============================================================
# Ce script construit la frontiere efficiente de Markowitz
# et trace la Capital Market Line (CML) qui relie le taux
# sans risque au portefeuille tangent optimal.
#
# CML : E(Rp) = Rf + [(Rm - Rf) / sigma_m] x sigma_p
# La pente de la CML = Sharpe ratio du portefeuille tangent
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

prix       = data["Close"]
rendements = prix.pct_change().dropna()

print("=" * 55)
print("   CAPITAL MARKET LINE — MARKOWITZ + CML")
print("=" * 55)
print(f"\nPeriode      : 2020-01-01 -> 2024-12-31")
print(f"Observations : {len(rendements)} jours")
print(f"Actifs       : {tickers}")

# ============================================================
# 2. PARAMETRES ANNUALISES
# ============================================================

mu_actifs  = rendements.mean() * 252
cov_matrix = rendements.cov()  * 252
n          = len(tickers)
rf         = 0.02  # taux sans risque 2%

# ============================================================
# 3. OPTIMISATION — PORTEFEUILLE TANGENT
# ============================================================

# On maximise le Sharpe en minimisant son negatif
# Contrainte : somme des poids = 1
# Bornes     : poids entre 0 et 1 (pas de vente a decouvert)
def neg_sharpe(w):
    rp = w @ mu_actifs
    vp = np.sqrt(w @ cov_matrix @ w)
    return -(rp - rf) / vp

contraintes = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
bornes      = [(0, 1)] * n
w0          = np.ones(n) / n

result    = minimize(neg_sharpe, w0, method="SLSQP",
                     bounds=bornes, constraints=contraintes)
w_optimal = result.x

# Metriques du portefeuille tangent
rp_opt     = w_optimal @ mu_actifs
vp_opt     = np.sqrt(w_optimal @ cov_matrix @ w_optimal)
sharpe_opt = (rp_opt - rf) / vp_opt

print("\n--- Portefeuille tangent ---")
for i, ticker in enumerate(tickers):
    print(f"  {ticker:8s} : {w_optimal[i]*100:.2f}%")
print(f"\n  Rendement  : {float(rp_opt)*100:.2f}%")
print(f"  Volatilite : {float(vp_opt)*100:.2f}%")
print(f"  Sharpe     : {float(sharpe_opt):.4f}")

# ============================================================
# 4. CAPITAL MARKET LINE
# ============================================================

# La CML est une droite qui part de Rf et passe par le
# portefeuille tangent. Sa pente = Sharpe ratio.
# En dessous du tangent : combinaison Rf + portefeuille tangent
# Au dessus du tangent  : effet de levier (emprunt a Rf)
sigma_range    = np.linspace(0, 0.30, 100)
cml_rendements = rf + (float(rp_opt) - rf) / float(vp_opt) * sigma_range

print(f"\n--- Capital Market Line ---")
print(f"  Pente CML (Sharpe) : {float(sharpe_opt):.4f}")
print(f"  Interpretation     : pour chaque 1% de risque accepte,")
print(f"                       le marche remunere {float(sharpe_opt)*100:.2f}% de rendement")

# ============================================================
# 5. SIMULATION 5000 PORTEFEUILLES
# ============================================================

# Ces portefeuilles servent UNIQUEMENT a visualiser la forme
# de la frontiere efficiente — ils ne sont pas utilises
# dans les calculs de la CML ni du portefeuille optimal.
rendements_port  = []
volatilites_port = []
sharpes_port     = []

for _ in range(5000):
    w      = np.random.dirichlet(np.ones(n))
    rp     = w @ mu_actifs
    vp     = np.sqrt(w @ cov_matrix @ w)
    sharpe = (rp - rf) / vp
    rendements_port.append(rp)
    volatilites_port.append(vp)
    sharpes_port.append(sharpe)

rendements_port  = np.array(rendements_port)
volatilites_port = np.array(volatilites_port)
sharpes_port     = np.array(sharpes_port)

# ============================================================
# 6. GRAPHIQUE — FRONTIERE EFFICIENTE + CML
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle(
    "Capital Market Line + Frontiere efficiente de Markowitz\n"
    "TTE.PA | MC.PA | OR.PA | CAC40 — 2020-2024",
    fontsize=13
)

# Nuage de 5000 portefeuilles colores par Sharpe
# Jaune = Sharpe eleve / Bleu = Sharpe faible
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

# CML — droite rouge pointillee
# C'est l'element central de ce script
ax.plot(
    sigma_range, cml_rendements,
    color="red", linewidth=2.5,
    linestyle="--",
    label=f"CML (pente = Sharpe = {float(sharpe_opt):.2f})",
    zorder=4
)

# Portefeuille tangent — etoile orange
# Point de tangence entre la CML et la frontiere efficiente
ax.scatter(
    float(vp_opt), float(rp_opt),
    color="orange", s=300,
    zorder=5, marker="*",
    label=f"Portefeuille tangent (Sharpe={float(sharpe_opt):.2f})"
)

# Taux sans risque — point vert sur l'axe Y
# Point de depart de la CML a sigma = 0
ax.scatter(
    0, rf,
    color="green", s=150,
    zorder=5, marker="D",
    label=f"Taux sans risque Rf = {rf*100:.1f}%"
)

ax.set_xlabel("Volatilite annualisee (risque)", fontsize=11)
ax.set_ylabel("Rendement annualise", fontsize=11)
ax.set_title(
    "5000 portefeuilles simules via Dirichlet + CML optimale",
    fontsize=10
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cml.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegarde : cml.png")
plt.show()