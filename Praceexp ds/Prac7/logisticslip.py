import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = (iris.target == 2).astype(int)  # Changed to make it a binary classification problem (Setosa vs Others)

# Selecting features and target
X = df_iris[['petal length (cm)', 'petal width (cm)']]  # Changed to use petal features for better classification
y = df_iris['target']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualizing Decision Boundary
plt.scatter(X_test['petal length (cm)'], X_test['petal width (cm)'], c=y_test, cmap='coolwarm', edgecolors='k', label='Actual')
plt.scatter(X_test['petal length (cm)'], X_test['petal width (cm)'], c=y_pred, cmap='coolwarm', alpha=0.5, marker='s', label='Predicted')
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()

