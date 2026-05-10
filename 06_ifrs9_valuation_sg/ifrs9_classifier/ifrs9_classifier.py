# ============================================================
# Day One PnL Calculator
# Contexte : Valuation Group - Société Générale
# Auteur   : Raphaël Noubiengang Tidjo
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
        'pnl_total'      : différence entre juste valeur et prix transaction
        'pnl_immediat'   : part reconnue immédiatement en résultat
        'reserve'        : part différée (non-observable)
    """
    # Validation 1 — tu l'as déjà écrite
    if not (0 <= part_observable <= 1):
        raise ValueError('verifier bien les valeurs svp')

# Validation 2 — prix_transaction positif
# même structure que validation 1
    if prix_transaction < 0:
              raise ValueError('verifier les valeurs svp')

# Validation 3 — juste_valeur positive
# même structure
    if juste_valeur < 0:
            raise ValueError('verifier les valeurs svp')
    
    # ÉTAPE 1 — Calcule le PnL total
    # (c'est la différence entre quoi et quoi ?)
    pnl_total =  prix_transaction-juste_valeur 
    
    # ÉTAPE 2 — Calcule la part reconnue immédiatement
    # (c'est quoi × quoi ?)
    pnl_immediat = pnl_total *part_observable
    
    # ÉTAPE 3 — Calcule la réserve
    # (c'est quoi - quoi ?)
    reserve =pnl_total*(1-part_observable) 
    
    return {
        'pnl_total'   : pnl_total,
        'pnl_immediat': pnl_immediat,
        'reserve'     : reserve
    }
def amortir_reserve(reserve_initiale, duree_vie):
    """
    Calcule le tableau d'amortissement linéaire
    de la réserve Day One PnL.

    Paramètres :
    ------------
    reserve_initiale : float — Montant de la réserve (en €)
    duree_vie        : int   — Durée de vie en années

    Retourne :
    ----------
    list de dict — une ligne par année avec :
        'annee'            : numéro de l'année
        'montant_libere'   : montant libéré cette année
        'reserve_restante' : réserve restante après libération
    """

    # ÉTAPE 1 — Calcule le montant libéré chaque année
    amortissement_annuel = reserve_initiale/duree_vie

    # ÉTAPE 2 — Crée une liste vide
    tableau = []

    # ÉTAPE 3 — Boucle sur chaque année
    for annee in range(1,duree_vie+1):

        # ÉTAPE 4 — Calcule la réserve restante
        # en utilisant la Façon A que tu as choisie
        reserve_restante = reserve_initiale - annee*amortissement_annuel

        # ÉTAPE 5 — Ajoute la ligne au tableau
        tableau.append({
            'annee'           : annee,
            'montant_libere'  : amortissement_annuel,
            'reserve_restante': reserve_restante
        })

    return tableau
def afficher_rapport(prix_transaction, juste_valeur, 
                     part_observable, duree_vie):
    """
    Affiche le rapport complet Day One PnL.
    """

    # ÉTAPE 1 — Appelle calculer_day_one_pnl()
    # pour récupérer les résultats
    resultat = calculer_day_one_pnl(prix_transaction, juste_valeur, part_observable)

    # ÉTAPE 2 — Appelle amortir_reserve()
    # pour récupérer le tableau
    tableau = amortir_reserve(resultat['reserve'], duree_vie)

    # ÉTAPE 3 — Affiche l'en-tête
    print("=" * 50)
    print("   DAY ONE PNL — VALUATION GROUP — SG")
    print("=" * 50)

    # ÉTAPE 4 — Affiche les résultats principaux
    # avec print() et les valeurs du dict resultat
    print(f"Prix transaction      : {prix_transaction:>12,.0f}€")
    print(f"Juste valeur modèle   : {juste_valeur:>12,.0f}€")
    print(f"PnL total             : {resultat['pnl_total']:>12,.0f}€")
    print(f"Part reconnue (imméd) : {resultat['pnl_immediat']:>12,.0f}€")
    print(f"Réserve différée      : {resultat['reserve']:>12,.0f}€")

    # ÉTAPE 5 — Affiche le tableau d'amortissement
    print(f"\n{'Année':>6} | {'Libéré':>12} | {'Réserve restante':>16}")
    print("-" * 42)

    for ligne in tableau:
        print(
            f"{ligne['annee']:>6} | "
            f"{ligne['montant_libere']:>11,.0f}€ | "
            f"{ligne['reserve_restante']:>15,.0f}€"
        )

    print("=" * 50)   
     