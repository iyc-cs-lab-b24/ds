import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating housing dataset
housing_data = {
    "HouseSize": [1500, 1800, 2400, 1400, 2800, 3000, 2200, 2600, 1600, 2000],
    "NumBedrooms": [3, 4, 3, 2, 5, 4, 3, 4, 2, 3],
    "NumBathrooms": [2, 3, 2, 1, 4, 3, 2, 3, 1, 2],
    "Price": [300000, 340000, 420000, 280000, 500000, 550000, 400000, 460000, 320000, 380000]
}

df_housing = pd.DataFrame(housing_data)

# Splitting into Features and Target
X = df_housing.drop(columns=["Price"])
y = df_housing["Price"]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-Squared Score:", r2_score(y_test, y_pred))

# Visualizing Actual vs Predicted Prices
plt.scatter(y_test, y_pred, color="blue", label="Predicted Prices")
plt.plot(y_test, y_test, color="red", linestyle="--", label="Actual Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()

