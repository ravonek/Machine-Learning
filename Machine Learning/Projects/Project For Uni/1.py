from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

# Удаление выбросов по весу
lower = df_encoded["weight_in_kg"].quantile(0.01)
upper = df_encoded["weight_in_kg"].quantile(0.99)
df_filtered = df_encoded[(df_encoded["weight_in_kg"] >= lower) & (df_encoded["weight_in_kg"] <= upper)]

X_reg = df_filtered.drop(columns=["weight_in_kg", "error", "error_type"])
y_reg = df_filtered["weight_in_kg"]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train, y_train)
y_pred = model_reg.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
