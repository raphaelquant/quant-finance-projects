# ============================================================
# Day One PnL Calculator
# Contexte : Valuation Group - Société Générale
# Auteur   : Raphaël Noubiengang Tidjo
# ============================================================
#
# Contexte métier :
# Quand SG vend un produit structuré, la différence entre
# le prix payé par le client et la juste valeur interne
# constitue le Day One PnL.
# Si ce PnL provient de paramètres non-observables (IFRS 13),
# il est différé en réserve et amorti progressivement.
# C'est le Valuation Group qui certifie ce calcul.
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# FONCTION 1 — CALCUL DU DAY ONE PNL
# ============================================================

def calculer_day_one_pnl(prix_transaction, juste_valeur, part_observable):
    """
    Calcule le Day One PnL d'un produit structuré selon IFRS 13.

    Paramètres :
    ------------
    prix_transaction : float — Ce que le client a payé (en €)
    juste_valeur     : float — Valorisation interne de SG (en €)
    part_observable  : float — Part des paramètres observables (entre 0 et 1)

    Retourne :
    ----------
    dict avec :
        'pnl_total'   : différence entre prix transaction et juste valeur
        'pnl_immediat': part reconnue immédiatement en résultat
        'reserve'     : part différée (paramètres non-observables)

    Exemple :
    ---------
    Prix client = 1 000 000€, Juste valeur = 940 000€
    → Day One PnL = +60 000€ (profit pour SG)
    → Part observable 60% → 36 000€ reconnus immédiatement
    → Réserve 40% → 24 000€ différés
    """

    # Validations des entrées
    if prix_transaction < 0:
        raise ValueError(
            f"prix_transaction ne peut pas être négatif, reçu : {prix_transaction}"
        )
    if juste_valeur < 0:
        raise ValueError(
            f"juste_valeur ne peut pas être négative, reçu : {juste_valeur}"
        )
    if not (0 <= part_observable <= 1):
        raise ValueError(
            f"part_observable doit être entre 0 et 1, reçu : {part_observable}"
        )

    # Calcul du PnL total
    # Si positif → SG a vendu plus cher que sa juste valeur → profit
    # Si négatif → SG a vendu moins cher que sa juste valeur → perte
    pnl_total = prix_transaction - juste_valeur

    # Part reconnue immédiatement en résultat
    # (paramètres observables = vérifiables sur le marché)
    pnl_immediat = pnl_total * part_observable

    # Part différée en réserve
    # (paramètres non-observables = modèles internes SG)
    reserve = pnl_total * (1 - part_observable)

    return {
        'pnl_total'   : round(pnl_total, 2),
        'pnl_immediat': round(pnl_immediat, 2),
        'reserve'     : round(reserve, 2)
    }


# ============================================================
# FONCTION 2 — AMORTISSEMENT DE LA RÉSERVE
# ============================================================

def amortir_reserve(reserve_initiale, duree_vie):
    """
    Calcule le tableau d'amortissement linéaire de la réserve Day One PnL.

    La réserve est libérée progressivement sur la durée de vie du produit.
    En pratique, la libération peut être accélérée si l'IPV atteste
    que des paramètres non-observables sont devenus observables.

    Paramètres :
    ------------
    reserve_initiale : float — Montant de la réserve à amortir (en €)
    duree_vie        : int   — Durée de vie du produit en années

    Retourne :
    ----------
    list de dict — une ligne par année avec :
        'annee'            : numéro de l'année
        'montant_libere'   : montant libéré cette année (en €)
        'reserve_restante' : réserve restante après libération (en €)
    """

    # Protection contre la division par zéro
    if duree_vie == 0:
        raise ValueError("duree_vie ne peut pas être 0 — division impossible")
    if duree_vie < 0:
        raise ValueError(
            f"duree_vie ne peut pas être négative, reçu : {duree_vie}"
        )

    # Amortissement linéaire — même montant libéré chaque année
    # Choix de la formule fermée : reserve_restante = reserve_initiale - annee × amortissement
    # Avantage : on peut calculer n'importe quelle année directement
    amortissement_annuel = reserve_initiale / duree_vie

    tableau = []

    for annee in range(1, duree_vie + 1):
        reserve_restante = reserve_initiale - annee * amortissement_annuel

        tableau.append({
            'annee'           : annee,
            'montant_libere'  : round(amortissement_annuel, 2),
            'reserve_restante': round(reserve_restante, 2)
        })

    return tableau


# ============================================================
# FONCTION 3 — RAPPORT COMPLET
# ============================================================

def afficher_rapport(prix_transaction, juste_valeur,
                     part_observable, duree_vie):
    """
    Génère le rapport complet Day One PnL :
    - Résultats chiffrés
    - Tableau d'amortissement
    - Graphique A : évolution de la réserve
    - Graphique B : répartition PnL immédiat vs Réserve
    """

    # Calculs
    resultat = calculer_day_one_pnl(prix_transaction, juste_valeur, part_observable)
    tableau  = amortir_reserve(resultat['reserve'], duree_vie)

    # --------------------------------------------------------
    # RAPPORT TEXTE
    # --------------------------------------------------------
    print("=" * 55)
    print("   DAY ONE PNL — VALUATION GROUP — SOCIÉTÉ GÉNÉRALE")
    print("=" * 55)
    print(f"Prix transaction      : {prix_transaction:>15,.0f} €")
    print(f"Juste valeur modèle   : {juste_valeur:>15,.0f} €")
    print("-" * 55)
    print(f"PnL total             : {resultat['pnl_total']:>15,.0f} €")
    print(f"Part reconnue (imméd) : {resultat['pnl_immediat']:>15,.0f} €  "
          f"({part_observable:.0%} observable)")
    print(f"Réserve différée      : {resultat['reserve']:>15,.0f} €  "
          f"({1-part_observable:.0%} non-observable)")
    print("=" * 55)

    print(f"\n{'Année':>6} | {'Libéré':>14} | {'Réserve restante':>16}")
    print("-" * 44)

    for ligne in tableau:
        print(
            f"{ligne['annee']:>6} | "
            f"{ligne['montant_libere']:>13,.0f}€ | "
            f"{ligne['reserve_restante']:>15,.0f}€"
        )

    print("=" * 55)

    # --------------------------------------------------------
    # GRAPHIQUE A — Évolution de la réserve dans le temps
    # --------------------------------------------------------
    annees   = [ligne['annee'] for ligne in tableau]
    reserves = [ligne['reserve_restante'] for ligne in tableau]
    liberes  = [ligne['montant_libere'] for ligne in tableau]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Day One PnL — Valuation Group — Société Générale",
        fontsize=13, fontweight='bold'
    )

    # Graphique A — Réserve restante
    ax1 = axes[0]
    ax1.plot(
        annees, reserves,
        color='#e74c3c', marker='o',
        linewidth=2, markersize=8,
        label='Réserve restante'
    )
    ax1.fill_between(annees, reserves, alpha=0.15, color='#e74c3c')
    ax1.bar(
        annees, liberes,
        alpha=0.4, color='#2ecc71',
        label='Montant libéré', width=0.4
    )
    ax1.set_title("Évolution de la réserve et libérations annuelles",
                  fontweight='bold')
    ax1.set_xlabel("Années")
    ax1.set_ylabel("Montant (€)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Ajout valeurs sur les points
    for a, r in zip(annees, reserves):
        ax1.annotate(
            f"{r:,.0f}€",
            xy=(a, r),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center', fontsize=8
        )

    # --------------------------------------------------------
    # GRAPHIQUE B — Répartition PnL immédiat vs Réserve
    # --------------------------------------------------------
    ax2 = axes[1]

    valeurs  = [resultat['pnl_immediat'], resultat['reserve']]
    labels   = [
        f"PnL immédiat\n{resultat['pnl_immediat']:,.0f}€",
        f"Réserve différée\n{resultat['reserve']:,.0f}€"
    ]
    couleurs = ['#2ecc71', '#e74c3c']
    explode  = (0.05, 0.05)

    ax2.pie(
        valeurs,
        labels     = labels,
        colors     = couleurs,
        autopct    = '%1.1f%%',
        startangle = 90,
        explode    = explode,
        shadow     = True
    )
    ax2.set_title(
        "Répartition PnL immédiat vs Réserve différée",
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig('day_one_pnl_rapport.png', dpi=150, bbox_inches='tight')
    print("\nGraphique sauvegardé : day_one_pnl_rapport.png")
    plt.show()


# ============================================================
# DÉMONSTRATION
# ============================================================

if __name__ == "__main__":

    # Exemple 1 — Profit pour SG
    print("\n=== EXEMPLE 1 — PROFIT POUR SG ===")
    afficher_rapport(
        prix_transaction = 1_000_000,
        juste_valeur     =   940_000,
        part_observable  =       0.6,
        duree_vie        =         4
    )

    # Exemple 2 — Perte pour SG
    print("\n=== EXEMPLE 2 — PERTE POUR SG ===")
    afficher_rapport(
        prix_transaction =   800_000,
        juste_valeur     =   850_000,
        part_observable  =       0.5,
        duree_vie        =         3
    )