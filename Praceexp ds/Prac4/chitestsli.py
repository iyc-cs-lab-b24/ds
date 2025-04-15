import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Employee aptitude and job proficiency data
data = {
    "aptitude": [85, 65, 50, 68, 87, 74, 65, 96, 68, 94, 73, 84, 85, 87, 91],
    "jobprof": [70, 90, 80, 89, 88, 86, 78, 67, 86, 90, 92, 94, 99, 93, 87]
}
df = pd.DataFrame(data)

# Convert continuous variables into categorical bins
df['aptitude_cat'] = pd.qcut(df['aptitude'], q=3, labels=['Low', 'Medium', 'High'])
df['jobprof_cat'] = pd.qcut(df['jobprof'], q=3, labels=['Low', 'Medium', 'High'])

# Create a contingency table
contingency_table = pd.crosstab(df['aptitude_cat'], df['jobprof_cat'])

# Perform Chi-Square Test
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Print results
print(f"Chi-Square Statistic: {chi2_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the Null Hypothesis: There is a significant correlation between aptitude and job proficiency.")
else:
    print("Fail to Reject the Null Hypothesis: No significant correlation found.")

