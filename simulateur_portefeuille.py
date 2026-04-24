# ============================================================
# Simulateur de Portefeuille 2 Actifs
# Concepts : Rendement, Volatilité, Ratio de Sharpe
# Auteur   : NOUBIENGANG TIDJO  RAPHAEL
# ============================================================

import math  # Pour la fonction racine carrée (sqrt)

# ============================================================
# 1. PARAMETRES D'ENTREE
# ============================================================

# Rendements espérés annuels de chaque actif
r1 = 0.08   # Actif 1 : 8% de rendement attendu par an
r2 = 0.05   # Actif 2 : 5% de rendement attendu par an

# Volatilités (écarts-types) annuelles de chaque actif
# Mesure le risque individuel de chaque actif
std1 = 0.20  # Actif 1 : 20% de volatilité (plus risqué)
std2 = 0.10  # Actif 2 : 10% de volatilité (moins risqué)

# Corrélation entre les deux actifs (entre -1 et +1)
# -1 : évoluent en sens opposé (diversification maximale)
#  0 : évoluent indépendamment
# +1 : évoluent dans le même sens (aucune diversification)
corr = -0.3  # Légèrement inversés → bonne diversification

# Pondérations des actifs dans le portefeuille
# Règle absolue : w1 + w2 = 1 (100% du capital investi)
w1 = 0.60   # 60% du capital dans l'actif 1
w2 = 0.40   # 40% du capital dans l'actif 2

# Taux sans risque (ex: obligations d'Etat)
# Référence pour calculer la prime de risque
rf = 0.02   # 2% par an

# ============================================================
# 2. CALCULS
# ============================================================

# --- Rendement du portefeuille ---
# Moyenne pondérée des rendements individuels
# Si w1 augmente → rp se rapproche de r1
rp = w1 * r1 + w2 * r2

# --- Volatilité du portefeuille ---
# Formule issue de la théorie de Markowitz
# Le 3ème terme (2*w1*w2*corr*std1*std2) est l'effet diversification
# Si corr < 0 → ce terme est négatif → réduit la volatilité totale
# C'est mathématiquement pourquoi diversifier réduit le risque
vp = math.sqrt(
    (w1**2) * (std1**2) +          # Risque pur actif 1
    (w2**2) * (std2**2) +          # Risque pur actif 2
    2 * w1 * w2 * corr * std1 * std2  # Effet de diversification
)

# --- Ratio de Sharpe ---
# Mesure le rendement EXCEDENTAIRE par unité de risque prise
# Numérateur   : ce qu'on gagne EN PLUS du taux sans risque
# Dénominateur : le risque pris pour y arriver
# Plus le Sharpe est élevé, plus le portefeuille est efficace
# Règle : Sharpe > 1 = bon / > 2 = excellent / < 0 = catastrophique
sharpe = (rp - rf) / vp

# ============================================================
# 3. AFFICHAGE DES RESULTATS
# ============================================================

print("=" * 50)
print("   SIMULATEUR DE PORTEFEUILLE — 2 ACTIFS")
print("=" * 50)

print("\n--- Paramètres ---")
print(f"Rendement Actif 1     : {r1*100:.1f}%")
print(f"Rendement Actif 2     : {r2*100:.1f}%")
print(f"Volatilité Actif 1    : {std1*100:.1f}%")
print(f"Volatilité Actif 2    : {std2*100:.1f}%")
print(f"Corrélation           : {corr}")
print(f"Pondération w1 / w2   : {w1*100:.0f}% / {w2*100:.0f}%")
print(f"Taux sans risque      : {rf*100:.1f}%")

print("\n--- Résultats ---")
print(f"Rendement portefeuille : {rp*100:.2f}%")
print(f"Volatilité portefeuille: {vp*100:.2f}%")
print(f"Ratio de Sharpe        : {sharpe:.4f}")

print("\n--- Interprétation Sharpe ---")
if sharpe < 0:
    print("Sharpe < 0  : Catastrophique — le sans-risque est meilleur")
elif sharpe < 0.5:
    print("Sharpe < 0.5 : Médiocre — risque mal rémunéré")
elif sharpe < 1:
    print("Sharpe < 1   : Acceptable — peut être amélioré")
elif sharpe < 2:
    print("Sharpe < 2   : Bon — portefeuille efficace")
else:
    print("Sharpe > 2   : Excellent — très rare en pratique")

print("=" * 50)