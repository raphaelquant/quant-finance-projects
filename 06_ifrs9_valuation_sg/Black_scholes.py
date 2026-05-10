# ============================================================
# Black-Scholes Pricer — Call, Put, Delta & IPV Checker
# Contexte : Valuation Group - Société Générale
# Auteur   : Raphaël Noubiengang Tidjo
# ============================================================
#
# Contexte métier :
# Le Valuation Group utilise ce pricer pour :
# 1. Calculer le prix théorique des options européennes
# 2. Calculer le Delta pour le hedging
# 3. Vérifier la parité call-put (IPV)
# 4. Détecter les incohérences dans les prix du Front Office
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# FONCTION 1 — CALCUL DE D1 ET D2
# ============================================================

def calculer_d1_d2(S, K, r, T, sigma):
    """
    Calcule d1 et d2 de la formule Black-Scholes.

    Paramètres :
    ------------
    S     : float — Prix actuel du sous-jacent (€)
    K     : float — Prix d'exercice strike (€)
    r     : float — Taux sans risque annuel
    T     : float — Durée jusqu'à maturité en années
    sigma : float — Volatilité annuelle du sous-jacent

    Retourne :
    ----------
    tuple (d1, d2)
    """

    # Validations
    if S <= 0:
        raise ValueError(f"S doit être strictement positif, reçu : {S}")
    if K <= 0:
        raise ValueError(f"K doit être strictement positif, reçu : {K}")
    if r < 0:
        raise ValueError(f"r ne peut pas être négatif, reçu : {r}")
    if T <= 0:
        raise ValueError(f"T doit être strictement positif, reçu : {T}")
    if sigma <= 0:
        raise ValueError(f"sigma doit être strictement positif, reçu : {sigma}")

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2


# ============================================================
# FONCTION 2 — PRIX DU CALL
# ============================================================

def black_scholes_call(S, K, r, T, sigma):
    """
    Calcule le prix d'un call européen selon Black-Scholes.

    Formule : C = S×N(d1) - K×e^(-rT)×N(d2)

    Paramètres :
    ------------
    S     : float — Prix actuel du sous-jacent (€)
    K     : float — Prix d'exercice strike (€)
    r     : float — Taux sans risque annuel
    T     : float — Durée jusqu'à maturité en années
    sigma : float — Volatilité annuelle du sous-jacent

    Retourne :
    ----------
    float — Prix du call (€)
    """

    d1, d2 = calculer_d1_d2(S, K, r, T, sigma)
    call   = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return round(call, 4)


# ============================================================
# FONCTION 3 — PRIX DU PUT (PARITÉ CALL-PUT)
# ============================================================

def black_scholes_put(S, K, r, T, sigma):
    """
    Calcule le prix d'un put européen via la parité call-put.

    Formule : P = C - S + K×e^(-rT)

    Principe : deux façons différentes d'obtenir le même
    résultat futur doivent coûter le même prix aujourd'hui
    (absence d'arbitrage).

    Paramètres :
    ------------
    S     : float — Prix actuel du sous-jacent (€)
    K     : float — Prix d'exercice strike (€)
    r     : float — Taux sans risque annuel
    T     : float — Durée jusqu'à maturité en années
    sigma : float — Volatilité annuelle du sous-jacent

    Retourne :
    ----------
    float — Prix du put (€)
    """

    call = black_scholes_call(S, K, r, T, sigma)
    put  = call - S + K * np.exp(-r * T)

    return round(put, 4)


# ============================================================
# FONCTION 4 — DELTA DU CALL
# ============================================================

def calculer_delta(S, K, r, T, sigma):
    """
    Calcule le Delta du call.

    Delta = N(d1) = sensibilité du prix du call
    au prix du sous-jacent.

    Interprétation métier :
    Si Delta = 0.62 et on vend 100 calls
    → on achète 62 actions pour se couvrir (Delta hedging)

    Retourne :
    ----------
    float — Delta entre 0 et 1
    """

    d1, _ = calculer_d1_d2(S, K, r, T, sigma)
    delta = norm.cdf(d1)

    return round(delta, 4)


# ============================================================
# FONCTION 5 — DÉTECTION D'ARBITRAGE (IPV)
# ============================================================

def detecter_arbitrage(call_trader, put_trader,
                       S, K, r, T, sigma,
                       seuil=0.50):
    """
    Compare les prix du trader avec Black-Scholes.
    Détecte les incohérences suspectes — processus IPV.

    Dans le Valuation Group, cette fonction simule la
    vérification indépendante des prix (IPV) :
    on compare les prix déclarés par le Front Office
    avec les prix théoriques de Black-Scholes.

    Paramètres :
    ------------
    call_trader : float — Prix du call déclaré par le trader
    put_trader  : float — Prix du put déclaré par le trader
    seuil       : float — Écart maximum toléré (défaut 0.50€)

    Retourne :
    ----------
    dict avec écarts et alertes
    """

    # Prix théoriques selon Black-Scholes
    call_bs = black_scholes_call(S, K, r, T, sigma)
    put_bs  = black_scholes_put(S, K, r, T, sigma)

    # Écarts — abs() car on mesure une distance
    ecart_call = abs(call_trader - call_bs)
    ecart_put  = abs(put_trader  - put_bs)

    # Vérification parité call-put sur prix TRADER
    parite_trader = call_trader - put_trader
    parite_bs     = S - K * np.exp(-r * T)
    ecart_parite  = abs(parite_trader - parite_bs)

    # Alertes — True si écart dépasse le seuil
    alerte_call   = ecart_call   > seuil
    alerte_put    = ecart_put    > seuil
    alerte_parite = ecart_parite > seuil

    return {
        'call_trader'   : call_trader,
        'call_bs'       : call_bs,
        'ecart_call'    : round(ecart_call, 4),
        'alerte_call'   : alerte_call,
        'put_trader'    : put_trader,
        'put_bs'        : put_bs,
        'ecart_put'     : round(ecart_put, 4),
        'alerte_put'    : alerte_put,
        'ecart_parite'  : round(ecart_parite, 4),
        'alerte_parite' : alerte_parite,
    }


# ============================================================
# FONCTION 6 — RAPPORT IPV COMPLET
# ============================================================

def afficher_rapport_ipv(call_trader, put_trader,
                         S, K, r, T, sigma, seuil=0.50):
    """
    Affiche le rapport IPV complet avec graphiques.
    """

    result = detecter_arbitrage(
        call_trader, put_trader, S, K, r, T, sigma, seuil
    )
    delta = calculer_delta(S, K, r, T, sigma)

    # Rapport texte
    print("=" * 60)
    print("   RAPPORT IPV — VALUATION GROUP — SOCIÉTÉ GÉNÉRALE")
    print("=" * 60)
    print(f"Paramètres : S={S}€ | K={K}€ | r={r:.0%} | "
          f"T={T}an | σ={sigma:.0%}")
    print("-" * 60)
    print(f"{'':20} {'Trader':>10} {'BS':>10} {'Écart':>10} {'Alerte':>8}")
    print("-" * 60)

    def statut(alerte):
        return "⚠️  OUI" if alerte else "✅ NON"

    print(f"{'Call':<20} {result['call_trader']:>10.4f} "
          f"{result['call_bs']:>10.4f} "
          f"{result['ecart_call']:>10.4f} "
          f"{statut(result['alerte_call']):>8}")

    print(f"{'Put':<20} {result['put_trader']:>10.4f} "
          f"{result['put_bs']:>10.4f} "
          f"{result['ecart_put']:>10.4f} "
          f"{statut(result['alerte_put']):>8}")

    print(f"{'Parité Call-Put':<20} {'':>10} {'':>10} "
          f"{result['ecart_parite']:>10.4f} "
          f"{statut(result['alerte_parite']):>8}")

    print("-" * 60)
    print(f"Delta du Call : {delta} "
          f"→ couvrir 100 calls = acheter {delta*100:.0f} actions")
    print("=" * 60)

    # Graphiques
    S_values = np.linspace(60, 140, 100)
    calls    = [black_scholes_call(s, K, r, T, sigma) for s in S_values]
    puts     = [black_scholes_put(s, K, r, T, sigma)  for s in S_values]
    deltas   = [calculer_delta(s, K, r, T, sigma)     for s in S_values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Black-Scholes — Valuation Group — Société Générale",
        fontsize=13, fontweight='bold'
    )

    # Graphique 1 — Call et Put
    axes[0].plot(S_values, calls, color='#2ecc71',
                 label='Call', linewidth=2)
    axes[0].plot(S_values, puts,  color='#e74c3c',
                 label='Put',  linewidth=2)
    axes[0].axvline(x=K, color='gray', linestyle='--',
                    label=f'Strike K={K}€')
    axes[0].axvline(x=S, color='blue', linestyle=':',
                    alpha=0.7, label=f'S actuel={S}€')
    axes[0].set_title("Prix Call et Put selon Black-Scholes",
                      fontweight='bold')
    axes[0].set_xlabel("Prix du sous-jacent S (€)")
    axes[0].set_ylabel("Prix de l'option (€)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Graphique 2 — Delta
    axes[1].plot(S_values, deltas, color='#3498db', linewidth=2)
    axes[1].axvline(x=K, color='gray', linestyle='--',
                    label=f'Strike K={K}€')
    axes[1].axhline(y=0.5, color='orange', linestyle='--',
                    label='Delta=0.5 (ATM)')
    axes[1].axvline(x=S, color='blue', linestyle=':',
                    alpha=0.7, label=f'S actuel={S}€')
    axes[1].set_title("Delta du Call", fontweight='bold')
    axes[1].set_xlabel("Prix du sous-jacent S (€)")
    axes[1].set_ylabel("Delta")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('black_scholes_rapport.png', dpi=150,
                bbox_inches='tight')
    print("\nGraphique sauvegardé : black_scholes_rapport.png")
    plt.show()


# ============================================================
# DÉMONSTRATION
# ============================================================

if __name__ == "__main__":

    # Paramètres de marché
    S     = 100   # Prix actuel du sous-jacent
    K     = 100   # Strike
    r     = 0.05  # Taux sans risque 5%
    T     = 1     # 1 an
    sigma = 0.20  # Volatilité 20%

    # Prix théoriques
    call  = black_scholes_call(S, K, r, T, sigma)
    put   = black_scholes_put(S, K, r, T, sigma)
    delta = calculer_delta(S, K, r, T, sigma)

    print(f"Call  : {call}€")
    print(f"Put   : {put}€")
    print(f"Delta : {delta}")

    # Vérification parité
    parite_gauche = call - put
    parite_droite = S - K * np.exp(-r * T)
    print(f"\nVérification parité call-put :")
    print(f"C - P         = {parite_gauche:.4f}")
    print(f"S - Ke^(-rT)  = {parite_droite:.4f}")
    print(f"Parité OK     : {abs(parite_gauche - parite_droite) < 0.001}")

    # Simulation IPV — le trader déclare des prix suspects
    print("\n--- SIMULATION IPV ---")
    afficher_rapport_ipv(
        call_trader = 15.00,   # trader survalue le call
        put_trader  =  5.57,
        S=S, K=K, r=r, T=T, sigma=sigma,
        seuil=0.50
    )