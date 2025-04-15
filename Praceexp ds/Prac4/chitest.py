import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# 📊 Sample Data: Observed Frequencies
#                | Recovered | Not Recovered |
# Medicine A     |    60     |       40      |
# Medicine B     |    45     |       55      |
observed = np.array([[60, 40], [45, 55]])

# ✅ Chi-Square Test
chi2, p_val, dof, expected = chi2_contingency(observed)
alpha = 0.05

# 📢 Print Results
print("\n📊 Chi-Square Test Results:")
print(f"Chi-Square Statistic: {chi2:.4f}\nP-Value: {p_val:.4f}")

if p_val < alpha:
    result = "❌ Null Hypothesis Rejected!\n✅ Conclusion: Medicine's  effect different."
else:
    result = "✅ Null Hypothesis Accepted!\n🔹 Conclusion: No significant difference."

print("\n" + result)  # Terminal output

# 📊 Bar Graph for Better Visualization
labels = ["Recovered", "Not Recovered"]
medicine_A = observed[0]
medicine_B = observed[1]

x = np.arange(len(labels))
width = 0.4

plt.bar(x - width/2, medicine_A, width, label="Medicine A", color="lightblue")
plt.bar(x + width/2, medicine_B, width, label="Medicine B", color="orange")

plt.xticks(x, labels)
plt.ylabel("Number of Patients")
plt.title("Medicine A vs Medicine B (Recovery Rates)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

# 📌 Show Result on Graph
plt.text(0.5, max(medicine_A.max(), medicine_B.max()) - 5, result, fontsize=10, color="red", ha="center")

plt.show()
