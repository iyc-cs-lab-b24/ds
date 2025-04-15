import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("Book1.csv")
print("Original Data:\n", df.head(), "\n")

# Step 2: Check for missing values
print("Check if dataset has null (missing) values:\n", df.isnull(), "\n")

# Step 3: Count missing values column-wise
print("Missing values column-wise:\n", df.isnull().sum(), "\n")

# Step 4: Total missing values in dataset
print("Total number of missing values:", df.isnull().sum().sum(), "\n")

# Step 5: Handle missing values using fillna()
df_fillna = df.fillna(value=0)
print("After handling missing values using fillna(0):\n", df_fillna.head(), "\n")
print("Missing values remaining:", df_fillna.isnull().sum().sum(), "\n")

# Step 6: Handle missing values using dropna()
df_dropna = df.dropna()
print("After handling missing values using dropna():\n", df_dropna, "\n")
print("Missing values remaining:", df_dropna.isnull().sum().sum(), "\n")

# Step 7: Handle missing values using replace()
df_replace = df.replace(to_replace=np.nan, value="abc")
print("After handling missing values using replace(np.nan, 'abc'):\n", df_replace, "\n")
print("Missing values remaining:", df_replace.isnull().sum().sum(), "\n")

# -------------------------
# Step 8: Detect and Handle Outliers
# -------------------------
print("Detecting Outliers using IQR method...\n")

# Loop through numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    print(f"Outliers in '{col}':\n", outliers[[col]], "\n")

# Optional: Remove outliers (create a cleaned DataFrame)
df_no_outliers = df.copy()
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_limit) & (df_no_outliers[col] <= upper_limit)]

print("Data after removing outliers:\n", df_no_outliers.head(), "\n")
