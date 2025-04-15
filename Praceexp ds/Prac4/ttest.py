import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Sample data: Fever reduction effect of Medicine A & B
np.random.seed(42)
A = np.random.normal(2.5, 0.5, 30)
B = np.random.normal(2.0, 0.5, 30)

# Perform t-test
t_stat, p_val = ttest_ind(A, B)
alpha = 0.05

# Print results in terminal
print("\n📊 Hypothesis Testing Results:")
print(f"T-Statistic: {t_stat:.4f}\nP-Value: {p_val:.4f}")

if p_val < alpha:
    result = "❌ Null Hypothesis Rejected!\n✅ Conclusion: Medicine A B are different hai."
else:
    result = "✅ Null Hypothesis Accepted!\n🔹 Conclusion: No significant difference."

print("\n" + result)  # Display result in terminal

# 📊 Simple Graph (Boxplot)
plt.boxplot([A, B], labels=["Medicine A", "Medicine B"], patch_artist=True)
plt.title("Medicine A vs Medicine B")
plt.ylabel("Fever Reduction (°C)")
plt.grid(True, linestyle="--", alpha=0.5)

# Show result on graph
plt.text(1.5, max(A.max(), B.max()), result, fontsize=10, color="red", ha="center")
plt.show()

