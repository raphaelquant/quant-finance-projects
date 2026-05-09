# ============================================================
# IFRS 9 Stage Classifier & ECL Calculator
# Contexte : Valuation Group - Société Générale
# Auteur   : Raphaël Noubiengang Tidjo
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# PARTIE 1 — CLASSIFICATION IFRS 9
# ============================================================

def classify_stage(PD, statut_paiements, degradation_significative, defaut_avere):
    """
    Classifie un instrument financier en Stage 1, 2 ou 3 selon IFRS 9.

    Paramètres :
    ------------
    PD                       : float  — Probability of Default (entre 0 et 1)
                               Probabilité qu'un client fasse défaut dans les 12 mois
    statut_paiements         : int    — Nombre de jours d'impayés
    degradation_significative: str    — 'oui' / 'non'
                               Indique si le risque de crédit s'est significativement
                               dégradé depuis l'origination du prêt
    defaut_avere             : str    — 'oui' / 'non'
                               Indique si la contrepartie est en défaut confirmé

    Retourne :
    ----------
    int — 1, 2 ou 3 selon le stage IFRS 9 :
        Stage 1 → Risque faible  : ECL sur 12 mois
        Stage 2 → Risque élevé  : ECL sur durée de vie (pas en défaut)
        Stage 3 → Défaut avéré  : ECL sur durée de vie (en défaut)
    """

    # Validation des entrées
    if defaut_avere not in ['oui', 'non']:
        raise ValueError(
            f"defaut_avere doit être 'oui' ou 'non', reçu : '{defaut_avere}'"
        )
    if degradation_significative not in ['oui', 'non']:
        raise ValueError(
            f"degradation_significative doit être 'oui' ou 'non', "
            f"reçu : '{degradation_significative}'"
        )
    if not (0 <= PD <= 1):
        raise ValueError(f"PD doit être entre 0 et 1, reçu : {PD}")
    if statut_paiements < 0:
        raise ValueError(f"statut_paiements ne peut pas être négatif, reçu : {statut_paiements}")

    # Règles de classification IFRS 9
    # Stage 3 — Défaut avéré OU plus de 90 jours d'impayés
    if defaut_avere == 'oui' or statut_paiements > 90:
        return 3

    # Stage 2 — Dégradation significative OU plus de 30 jours d'impayés
    elif statut_paiements > 30 or degradation_significative == 'oui':
        return 2

    # Stage 1 — Situation normale
    else:
        return 1


# ============================================================
# PARTIE 2 — CALCUL DE L'ECL (Expected Credit Loss)
# ============================================================

def calculer_ecl(PD, LGD, EAD, stage, maturite_annees=5):
    """
    Calcule l'Expected Credit Loss (ECL) selon le stage IFRS 9.

    Paramètres :
    ------------
    PD             : float — Probability of Default (entre 0 et 1)
    LGD            : float — Loss Given Default : perte en cas de défaut (entre 0 et 1)
                     Ex : LGD = 0.45 → on perd 45% de l'exposition si défaut
    EAD            : float — Exposure At Default : montant exposé au défaut (en €)
    stage          : int   — Stage IFRS 9 (1, 2 ou 3)
    maturite_annees: float — Durée restante du contrat en années (pour Stage 2 et 3)

    Retourne :
    ----------
    float — Montant de la provision ECL en euros
    """

    # Stage 1 → ECL sur 12 mois uniquement
    if stage == 1:
        ecl = PD * LGD * EAD

    # Stage 2 → ECL sur toute la durée de vie restante
    elif stage == 2:
        ecl = PD * LGD * EAD * maturite_annees

    # Stage 3 → ECL intégral (défaut quasi certain)
    elif stage == 3:
        ecl = LGD * EAD  # PD ≈ 1 en défaut avéré

    else:
        raise ValueError(f"Stage invalide : {stage}. Doit être 1, 2 ou 3.")

    return round(ecl, 2)


# ============================================================
# PARTIE 3 — RAPPORT DE PROVISIONNEMENT
# ============================================================

def generer_rapport(portefeuille):
    """
    Génère un rapport de provisionnement IFRS 9 pour un portefeuille d'instruments.

    Paramètre :
    -----------
    portefeuille : list de dict — chaque instrument contient :
        {
            'nom'                     : str,
            'PD'                      : float,
            'LGD'                     : float,
            'EAD'                     : float,
            'statut_paiements'        : int,
            'degradation_significative': str ('oui'/'non'),
            'defaut_avere'            : str ('oui'/'non'),
            'maturite_annees'         : float
        }

    Retourne :
    ----------
    list de dict — rapport enrichi avec stage et ECL pour chaque instrument
    """

    rapport = []
    total_ead  = 0
    total_ecl  = 0

    print("=" * 70)
    print("   RAPPORT IFRS 9 — VALUATION GROUP — SOCIÉTÉ GÉNÉRALE")
    print("=" * 70)
    print(f"{'Instrument':<20} {'Stage':<8} {'PD':>6} {'EAD':>12} {'ECL':>12} {'Taux prov.':>12}")
    print("-" * 70)

    for inst in portefeuille:
        stage = classify_stage(
            inst['PD'],
            inst['statut_paiements'],
            inst['degradation_significative'],
            inst['defaut_avere']
        )
        ecl = calculer_ecl(
            inst['PD'],
            inst['LGD'],
            inst['EAD'],
            stage,
            inst['maturite_annees']
        )
        taux_prov = (ecl / inst['EAD'] * 100) if inst['EAD'] > 0 else 0

        total_ead += inst['EAD']
        total_ecl += ecl

        rapport.append({
            **inst,
            'stage': stage,
            'ecl'  : ecl,
            'taux_provisionnement': round(taux_prov, 2)
        })

        stage_label = f"Stage {stage}"
        print(
            f"{inst['nom']:<20} {stage_label:<8} {inst['PD']:>6.0%} "
            f"{inst['EAD']:>12,.0f}€ {ecl:>12,.2f}€ {taux_prov:>11.2f}%"
        )

    print("-" * 70)
    taux_global = (total_ecl / total_ead * 100) if total_ead > 0 else 0
    print(f"{'TOTAL PORTEFEUILLE':<20} {'':8} {'':>6} "
          f"{total_ead:>12,.0f}€ {total_ecl:>12,.2f}€ {taux_global:>11.2f}%")
    print("=" * 70)

    # Répartition par stage
    stages = [r['stage'] for r in rapport]
    ecls   = [r['ecl'] for r in rapport]
    print(f"\nRépartition : "
          f"Stage 1 = {stages.count(1)} | "
          f"Stage 2 = {stages.count(2)} | "
          f"Stage 3 = {stages.count(3)}")
    print(f"ECL total   : {total_ecl:,.2f}€ / EAD total : {total_ead:,.0f}€")
    print(f"Taux global de provisionnement : {taux_global:.2f}%\n")

    return rapport


# ============================================================
# PARTIE 4 — VISUALISATION
# ============================================================

def visualiser_portefeuille(rapport):
    """
    Génère 3 graphiques professionnels pour le rapport IFRS 9.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "IFRS 9 — Rapport de Provisionnement | Valuation Group — Société Générale",
        fontsize=14, fontweight='bold', y=1.02
    )

    couleurs_stages = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}
    noms   = [r['nom']   for r in rapport]
    stages = [r['stage'] for r in rapport]
    ecls   = [r['ecl']   for r in rapport]
    eads   = [r['EAD']   for r in rapport]

    # --- Graphique 1 : Distribution des stages ---
    ax1 = axes[0]
    stage_counts = {1: stages.count(1), 2: stages.count(2), 3: stages.count(3)}
    bars = ax1.bar(
        [f"Stage {s}" for s in stage_counts.keys()],
        stage_counts.values(),
        color=[couleurs_stages[s] for s in stage_counts.keys()],
        edgecolor='white', linewidth=1.5
    )
    ax1.set_title("Distribution des Stages IFRS 9", fontweight='bold')
    ax1.set_ylabel("Nombre d'instruments")
    for bar, val in zip(bars, stage_counts.values()):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(val), ha='center', fontweight='bold'
        )
    ax1.set_ylim(0, max(stage_counts.values()) + 1)
    ax1.grid(axis='y', alpha=0.3)

    # --- Graphique 2 : ECL par instrument ---
    ax2 = axes[1]
    bar_colors = [couleurs_stages[s] for s in stages]
    bars2 = ax2.bar(noms, ecls, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax2.set_title("ECL par Instrument (€)", fontweight='bold')
    ax2.set_ylabel("Expected Credit Loss (€)")
    ax2.set_xticklabels(noms, rotation=30, ha='right', fontsize=9)
    for bar, val in zip(bars2, ecls):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ecls) * 0.01,
            f"{val:,.0f}€", ha='center', fontsize=8, fontweight='bold'
        )
    ax2.grid(axis='y', alpha=0.3)

    # --- Graphique 3 : Taux de provisionnement ---
    ax3 = axes[2]
    taux = [r['taux_provisionnement'] for r in rapport]
    bars3 = ax3.barh(noms, taux, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax3.set_title("Taux de Provisionnement (%)", fontweight='bold')
    ax3.set_xlabel("Taux de provisionnement (%)")
    for bar, val in zip(bars3, taux):
        ax3.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va='center', fontsize=9, fontweight='bold'
        )
    ax3.grid(axis='x', alpha=0.3)

    # Légende
    legende = [
        mpatches.Patch(color=couleurs_stages[1], label='Stage 1 — Risque faible'),
        mpatches.Patch(color=couleurs_stages[2], label='Stage 2 — Risque élevé'),
        mpatches.Patch(color=couleurs_stages[3], label='Stage 3 — Défaut avéré'),
    ]
    fig.legend(handles=legende, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.08), fontsize=10)

    plt.tight_layout()
    plt.savefig('ifrs9_rapport.png', dpi=150, bbox_inches='tight')
    print("Graphique sauvegardé : ifrs9_rapport.png")
    plt.show()


# ============================================================
# PARTIE 5 — DÉMONSTRATION COMPLÈTE
# ============================================================

if __name__ == "__main__":

    # --- Test du classifier de base (ton code original enrichi) ---
    print("\n--- TEST CLASSIFIER DE BASE ---\n")

    instrument1 = [0.98, 92, 'oui', 'oui']
    instrument2 = [0.7,  32, 'oui', 'non']
    instrument3 = [0.2,   0, 'non', 'non']

    for i, inst in enumerate([instrument1, instrument2, instrument3], 1):
        stage = classify_stage(inst[0], inst[1], inst[2], inst[3])
        print(f"Instrument {i} → Stage {stage}  "
              f"(PD={inst[0]:.0%}, Jours={inst[1]}, "
              f"Dégradation={inst[2]}, Défaut={inst[3]})")

    # --- Portefeuille complet avec ECL ---
    print("\n--- PORTEFEUILLE COMPLET ---\n")

    portefeuille = [
        {
            'nom'                      : 'Crédit Corp A',
            'PD'                       : 0.01,
            'LGD'                      : 0.45,
            'EAD'                      : 1_000_000,
            'statut_paiements'         : 0,
            'degradation_significative': 'non',
            'defaut_avere'             : 'non',
            'maturite_annees'          : 5
        },
        {
            'nom'                      : 'Crédit Corp B',
            'PD'                       : 0.08,
            'LGD'                      : 0.40,
            'EAD'                      : 500_000,
            'statut_paiements'         : 35,
            'degradation_significative': 'oui',
            'defaut_avere'             : 'non',
            'maturite_annees'          : 3
        },
        {
            'nom'                      : 'Crédit Corp C',
            'PD'                       : 0.45,
            'LGD'                      : 0.60,
            'EAD'                      : 750_000,
            'statut_paiements'         : 60,
            'degradation_significative': 'oui',
            'defaut_avere'             : 'non',
            'maturite_annees'          : 4
        },
        {
            'nom'                      : 'Crédit Corp D',
            'PD'                       : 0.98,
            'LGD'                      : 0.75,
            'EAD'                      : 300_000,
            'statut_paiements'         : 120,
            'degradation_significative': 'oui',
            'defaut_avere'             : 'oui',
            'maturite_annees'          : 2
        },
        {
            'nom'                      : 'Crédit Corp E',
            'PD'                       : 0.02,
            'LGD'                      : 0.35,
            'EAD'                      : 2_000_000,
            'statut_paiements'         : 5,
            'degradation_significative': 'non',
            'defaut_avere'             : 'non',
            'maturite_annees'          : 7
        },
    ]

    rapport = generer_rapport(portefeuille)
    visualiser_portefeuille(rapport)