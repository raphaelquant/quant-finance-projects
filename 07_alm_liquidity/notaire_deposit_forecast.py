# ============================================================
# Modèle de Prévision des Dépôts Notaires — ALM CDC
# Régression linéaire + backtesting MAE/RMSE
# Auteur : Raphaël
# ============================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)

# Données simulées — 36 mois
taux_credit = np.random.normal(loc=3, scale=0.5, size=36)
taux_chomage = np.random.normal(loc=7, scale=0.8, size=36)
depot_notaires = np.random.normal(loc=5, scale=1, size=36)

df = pd.DataFrame({
    "taux_credit": taux_credit,
    "taux_chomage": taux_chomage,
    "depot_notaires": depot_notaires
})

# Train / Test split
X = df[["taux_credit", "taux_chomage"]]
y = df["depot_notaires"]
X_train, y_train = X.iloc[:24], y.iloc[:24]
X_test, y_test = X.iloc[24:], y.iloc[24:]

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Backtesting
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE  : {mae:.4f} Mds €")
print(f"RMSE : {rmse:.4f} Mds €")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(range(12), y_test.values, color='blue', label='Réel')
plt.plot(range(12), y_pred, color='red', label='Prédit')
plt.title('Prévision des Dépôts Notaires — CDC ALM')
plt.xlabel('Mois')
plt.ylabel('Dépôts (Mds €)')
plt.legend()
plt.savefig('notaire_forecast.png')
plt.show()