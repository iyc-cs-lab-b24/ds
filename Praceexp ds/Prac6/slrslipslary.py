import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Part A: Linear Regression for Salary Prediction

# Creating dataset
data_salary = {
    "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000, 100000]
}
df_salary = pd.DataFrame(data_salary)

# Splitting into Features and Target
X = df_salary[['YearsExperience']]  # Changed feature to 'YearsExperience'
y = df_salary['Salary']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Model Evaluation
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-Squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.4f}")

# Visualizing Regression Line
plt.scatter(X_test, y_test, color='blue', label='Actual Salary')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Years of Experience")  # Updated label to match new feature
plt.ylabel("Salary")
plt.title("Linear Regression for Salary Prediction")
plt.legend()
plt.show()

