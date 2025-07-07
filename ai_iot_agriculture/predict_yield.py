import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Sample data: [moisture, temperature, light, pH]
X_train = np.array([
    [30, 25, 200, 6.5],
    [45, 28, 250, 6.8],
    [20, 22, 180, 5.9],
    [50, 30, 300, 7.0],
])

# Sample yields (tons per hectare)
y_train = np.array([3.0, 3.5, 2.8, 3.8])

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# New sensor data for prediction
new_data = np.array([[40, 27, 220, 6.7]])

# Predict yield
predicted_yield = model.predict(new_data)
print(f"Predicted yield: {predicted_yield[0]:.2f} tons/ha")
