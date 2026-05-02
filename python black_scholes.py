# ============================================================
# Modele Black-Scholes — Pricing d'options europeennes
# Auteur  : raphaelquant
# Date    : Avril 2026
# GitHub  : https://github.com/raphaelquant
# ============================================================
# Ce script implemente le modele de Black-Scholes (1973) pour
# le pricing d'options europeennes calls et puts, ainsi que
# le calcul du Delta — premiere des Grecques.
#
# Parametres du modele :
#   S     : Prix spot de l'actif sous-jacent
#   K     : Prix d'exercice (strike)
#   T     : Maturite en annees
#   r     : Taux d'interet sans risque
#   sigma : Volatilite annualisee de l'actif
# ============================================================

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================
# 1. MODELE BLACK-SCHOLES — PRICING CALL ET PUT
# ============================================================

def black_scholes(S, K, T, r, sigma, type):
    """
    Calcule le prix d'une option europeenne via Black-Scholes.

    Formule :
        d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        Call = S*N(d1) - K*e^(-rT)*N(d2)
        Put  = K*e^(-rT)*N(-d2) - S*N(-d1)

    N() = fonction de repartition de la loi normale standard
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if type == "call":
        # Le call vaut l'esperance actualisee du gain max(S-K, 0)
        prix = S*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
    elif type == "put":
        # Le put vaut l'esperance actualisee du gain max(K-S, 0)
        prix = K*np.exp(-r*T)*stats.norm.cdf(-d2) - S*stats.norm.cdf(-d1)

    return prix


# ============================================================
# 2. DELTA — SENSIBILITE AU PRIX SPOT
# ============================================================

def delta(S, K, T, r, sigma, type):
    """
    Calcule le Delta de l'option — premiere des Grecques.

    Delta mesure la sensibilite du prix de l'option au prix spot S.
    Si S monte de 1 euro, le prix de l'option change de Delta euros.

        Delta call = N(d1)       -> toujours entre 0 et 1
        Delta put  = N(d1) - 1  -> toujours entre -1 et 0

    Interpretation :
        Delta call = 0.63 signifie que si S monte de 1 euro,
        le call monte d'environ 0.63 euro.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    if type == "call":
        return stats.norm.cdf(d1)
    elif type == "put":
        return stats.norm.cdf(d1) - 1


# ============================================================
# 3. PARAMETRES DU MODELE
# ============================================================

# Parametres choisis pour modeliser une option sur un indice type CAC40
S     = 100    # Prix spot actuel de l'actif sous-jacent (en euros)
K     = 100    # Strike — option "at the money" (S = K)
T     = 1      # Maturite : 1 an
r     = 0.05   # Taux sans risque : 5% (taux OAT 10 ans approximatif)
sigma = 0.20   # Volatilite annualisee : 20% (volatilite historique CAC40)


# ============================================================
# 4. CALCUL DES PRIX ET DU DELTA
# ============================================================

prix_call  = black_scholes(S, K, T, r, sigma, "call")
prix_put   = black_scholes(S, K, T, r, sigma, "put")
delta_call = delta(S, K, T, r, sigma, "call")
delta_put  = delta(S, K, T, r, sigma, "put")

print("=" * 55)
print("   BLACK-SCHOLES — PRICING OPTIONS EUROPEENNES")
print("=" * 55)

print("\n--- Parametres ---")
print(f"  Prix spot (S)     : {S} EUR")
print(f"  Strike (K)        : {K} EUR")
print(f"  Maturite (T)      : {T} an")
print(f"  Taux sans risque  : {r*100:.1f}%")
print(f"  Volatilite sigma  : {sigma*100:.1f}%")

print("\n--- Prix des options ---")
print(f"  Call              : {prix_call:.4f} EUR")
print(f"  Put               : {prix_put:.4f} EUR")

# Verification de la parite call-put : C - P = S - K*exp(-rT)
parite = S - K*np.exp(-r*T)
print(f"\n--- Verification parite call-put ---")
print(f"  C - P             : {prix_call - prix_put:.4f} EUR")
print(f"  S - K*e(-rT)      : {parite:.4f} EUR")
print(f"  Ecart             : {abs((prix_call - prix_put) - parite):.6f} EUR (quasi nul)")

print("\n--- Delta (sensibilite au spot) ---")
print(f"  Delta call        : {delta_call:.4f}")
print(f"  Delta put         : {delta_put:.4f}")
print(f"  Interpretation    : si S monte de 1 EUR,")
print(f"                      le call monte de {delta_call:.4f} EUR")
print(f"                      le put baisse de {abs(delta_put):.4f} EUR")


# ============================================================
# 5. GRAPHIQUE — PRIX DU CALL EN FONCTION DU SPOT
# ============================================================

# On fait varier le prix spot de 60 a 140 pour voir
# comment le prix du call evolue autour du strike K = 100
spots = np.linspace(60, 140, 200)
calls = [black_scholes(s, K, T, r, sigma, "call") for s in spots]
puts  = [black_scholes(s, K, T, r, sigma, "put")  for s in spots]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Black-Scholes — Pricing d'options europeennes\n"
    f"K={K} | T={T}an | r={r*100:.0f}% | sigma={sigma*100:.0f}%",
    fontsize=13
)

# --- Graphique 1 : Prix du Call ---
axes[0].plot(spots, calls, color="#185FA5", linewidth=2, label="Prix du Call")
axes[0].axvline(x=S, color="red", linestyle="--", linewidth=1.2,
                label=f"Spot actuel S={S}")
axes[0].axvline(x=K, color="gray", linestyle=":", linewidth=1,
                label=f"Strike K={K}")
axes[0].set_title("Prix du Call en fonction du Spot")
axes[0].set_xlabel("Prix Spot (S)")
axes[0].set_ylabel("Prix du Call (EUR)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Graphique 2 : Prix du Put ---
axes[1].plot(spots, puts, color="#e67e22", linewidth=2, label="Prix du Put")
axes[1].axvline(x=S, color="red", linestyle="--", linewidth=1.2,
                label=f"Spot actuel S={S}")
axes[1].axvline(x=K, color="gray", linestyle=":", linewidth=1,
                label=f"Strike K={K}")
axes[1].set_title("Prix du Put en fonction du Spot")
axes[1].set_xlabel("Prix Spot (S)")
axes[1].set_ylabel("Prix du Put (EUR)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("black_scholes.png", dpi=150, bbox_inches="tight")
print("\nGraphique sauvegarde : black_scholes.png")
plt.show()