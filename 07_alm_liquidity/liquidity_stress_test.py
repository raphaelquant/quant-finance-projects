# ============================================================
# Liquidity Stress Test — ALM Caisse des Dépôts
# Simulation de 3 scénarios de crise de liquidité
# Auteur : Raphaël
# ============================================================

import numpy as np
from matplotlib import pyplot as plt

# Simulation des dépôts notaires sur 12 mois (loi normale)
np.random.seed(42)
tableau = np.random.normal(loc=5, scale=1, size=12)

# Paramètres de stress
choc_depot = 0.30
choc_actifs = 0.40
actifs_liquides_base = 8

# Scénario 1 : retrait 30% dépôts notaires
depots_scenario1 = tableau * (1 - choc_depot)
gap_scenario1 = actifs_liquides_base - depots_scenario1 * 0.20

# Scénario 2 : actifs liquides perdent 40%
depots_scenario2 = tableau
actifs_scenario2 = actifs_liquides_base * (1 - choc_actifs)
gap_scenario2 = actifs_scenario2 - depots_scenario2 * 0.20

# Scénario 3 : double choc
depots_scenario3 = tableau * (1 - choc_depot)
actifs_scenario3 = actifs_liquides_base * (1 - choc_actifs)
gap_scenario3 = actifs_scenario3 - depots_scenario3 * 0.20

# Visualisation
mois = range(1, 13)
plt.figure(figsize=(12, 6))
plt.plot(mois, gap_scenario1, color='blue', label='Scénario 1 : choc dépôts')
plt.plot(mois, gap_scenario2, color='orange', label='Scénario 2 : choc actifs')
plt.plot(mois, gap_scenario3, color='red', label='Scénario 3 : double choc')
plt.axhline(y=0, color='black', linestyle='--', label='Seuil critique')
plt.title('Stress Test de Liquidité — Dépôts Notaires CDC')
plt.xlabel('Mois')
plt.ylabel('Gap de liquidité (Mds €)')
plt.legend()
plt.savefig('liquidity_stress_test.png')
plt.show()