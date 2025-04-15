import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_breast_cancer

# 📌 Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 1 = Malignant, 0 = Benign

# ✂ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔥 Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# 🌳 Decision Tree Model
tree_model = DecisionTreeClassifier(max_depth=3)  # Limiting depth for simplicity
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# 📊 Performance Metrics Function
def evaluate_model(name, y_test, y_pred):
    print(f"🔹 {name} Performance:")
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"✅ Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"✅ Recall: {recall_score(y_test, y_pred):.4f}")
    print("-" * 30)

# 🏆 Evaluate Models
evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Decision Tree", y_test, y_pred_tree)

# 📈 Visualizing Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(tree_model, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()
