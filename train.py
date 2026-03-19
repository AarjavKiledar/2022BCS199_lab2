import pandas as pd
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv('dataset/winequality-red.csv', sep=';')

# 2. Pre-processing and Feature Selection
# EXP-03: Random Forest (Baseline)
X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the selected model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 5. Save outputs
os.makedirs('output', exist_ok=True)
metrics = {
    "MSE": round(mse, 4),
    "R2 Score": round(r2, 4)
}
with open('output/results.json', 'w') as f:
    json.dump(metrics, f)

joblib.dump(model, 'output/model.pkl')

# 6. Print metrics to standard output
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")