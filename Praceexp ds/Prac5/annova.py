import scipy.stats as stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 📊 Sample Data (Replace with actual data)
group1 = [23, 25, 29, 34, 30]
group2 = [19, 20, 22, 25, 24]
group3 = [15, 18, 20, 21, 17]
group4 = [28, 24, 26, 30, 29]

# Combine all data into a single array
all_data = np.array(group1 + group2 + group3 + group4)

# Create corresponding group labels
group_labels = (['Group1'] * len(group1) + 
                ['Group2'] * len(group2) + 
                ['Group3'] * len(group3) + 
                ['Group4'] * len(group4))

# ✅ Perform One-Way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3, group4)

# 📢 Print ANOVA results
print("📊 One-Way ANOVA Results:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# ✅ Perform Tukey-Kramer post-hoc test (if p < 0.05)
if p_value < 0.05:
    print("\n🔍 Significant difference found! Performing post-hoc analysis...")
    tukey_results = pairwise_tukeyhsd(all_data, group_labels)

    # 📢 Print Tukey-Kramer results
    print("\n📊 Tukey-Kramer Post-Hoc Test Results:")
    print(tukey_results)
else:
    print("\n✅ No significant difference found between groups.")


