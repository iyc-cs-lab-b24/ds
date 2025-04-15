import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing  
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt

# Load the iris dataset
iris = fetch_california_housing  ()
data = pd.DataFrame(data=np.c_[iris['data'],iris['target']], columns=iris['feature_names']+['target'])

#Extract numerical features
numerical_features=iris['feature_names']

#Separate features and target variable

X=data[numerical_features]
y=data['target']

#standardization
scaler_standard=StandardScaler()
X_standardized=scaler_standard.fit_transform(X)

#normalization (min-max scaling)

scaler_minmax=MinMaxScaler()
X_normalized=scaler_minmax.fit_transform(X)
#print(X_normalized)

#if you want to represent in graph
#visualize the original , standardized,and normalized features
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X.iloc[:,0],X.iloc[:, 1], c=y, cmap='viridis')
plt.title('Original Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(132)
plt.scatter(X_standardized[:,0], X_standardized[:, 1], c=y,cmap='viridis')
plt.title('Standardized Features')
plt.xlabel('Feature 1(Standardized )')
plt.ylabel('Feature 1(Standardized )')

plt.subplot(133)
plt.scatter(X_normalized[:,0], X_normalized[:, 1], c=y,cmap='viridis')
plt.title('Normalized Features')
plt.xlabel('Feature 1(Normalized)')
plt.ylabel('Feature 1(Normalized)')

plt.tight_layout()
plt.show()
