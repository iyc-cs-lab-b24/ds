import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import seaborn as sns

# Part A: Decision Tree on Titanic Dataset

# Load Titanic dataset
df_titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")  # Changed to load Titanic dataset

# Selecting relevant features
X = df_titanic[['Pclass', 'Age', 'SibSp', 'Parch']].fillna(df_titanic[['Pclass', 'Age', 'SibSp', 'Parch']].median())
y = df_titanic['Survived']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Decision Tree Model
tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(X_train, y_train)

# Visualizing Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(tree_model, feature_names=['Pclass', 'Age', 'SibSp', 'Parch'], class_names=['Not Survived', 'Survived'], filled=True)
plt.title("Decision Tree for Titanic Survival Prediction")
plt.show()

# Part B: Linear Regression for Weight Prediction

data_weight = {
    "Height": [151, 174, 138, 186, 128, 136, 179, 163, 152],
    "Weight": [63, 81, 56, 91, 47, 57, 76, 72, 62]
}
df_weight = pd.DataFrame(data_weight)

# Splitting into Features and Target
X = df_weight[['Height']]  # Changed feature to 'Height'
y = df_weight['Weight']

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
plt.scatter(X_test, y_test, color='blue', label='Actual Weight')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Height")  # Updated label to match new feature
plt.ylabel("Weight")
plt.title("Linear Regression for Weight Prediction")
plt.legend()
plt.show()

