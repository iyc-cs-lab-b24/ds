import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# 📌 Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# 🎯 Feature: Average Rooms | Target: House Price
X = df[['AveRooms']]
y = housing.target

# ✂ Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Predict
y_pred = model.predict(X_test)

# 📊 Model Evaluation
print(f"✅ Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
print(f"✅ R-Squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"✅ Intercept: {model.intercept_:.4f}")
print(f"✅ Coefficient: {model.coef_[0]:.4f}")

# 📈 Visualizing Regression Line
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Average Rooms")
plt.ylabel("House Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()


