import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# Sample data: Student Scores
np.random.seed(42)
student_scores = np.array([72, 88, 64, 74, 67, 79, 85, 75, 89, 77])

# Hypothesized population mean
hypothesized_mean = 70

# Perform one-sample t-test
t_stat, p_val = ttest_1samp(student_scores, hypothesized_mean)
alpha = 0.05

# Print results
print("\n📊 Hypothesis Testing Results:")
print(f"T-Statistic: {t_stat:.4f}\nP-Value: {p_val:.4f}")

if p_val < alpha:
    result = "❌ Null Hypothesis Rejected!\n✅ Conclusion: The mean of student scores is significantly different from the hypothesized mean."
else:
    result = "✅ Null Hypothesis Accepted!\n🔹 Conclusion: No significant difference between the mean of student scores and the hypothesized mean."

print("\n" + result)  # Display result in terminal

# 📊 Simple Graph (Boxplot)
plt.boxplot(student_scores, patch_artist=True)
plt.title("Student Scores Distribution")
plt.ylabel("Scores")
plt.grid(True, linestyle="--", alpha=0.5)

# Show result on graph
plt.text(1, student_scores.max(), result, fontsize=10, color="red", ha="center")
plt.show()
