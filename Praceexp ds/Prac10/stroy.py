import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dummy Data: Netflix ki movie categories aur unka view count
data = {
    "Category": ["Action", "Comedy", "Horror", "Romance", "Sci-Fi"],
    "Views (Million)": [120, 85, 95, 60, 75],
    "Avg Rating": [8.5, 7.8, 7.2, 8.0, 7.5]
}

df = pd.DataFrame(data)

# Figure aur subplots banayein (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1st Chart: Bar Chart (Movie Category vs Views)
sns.barplot(x="Category", y="Views (Million)", data=df, ax=axes[0], palette="coolwarm")
axes[0].set_title("Popular Movie Categories on Netflix")
axes[0].set_ylabel("Views (Million)")

# 2nd Chart: Scatter Plot (Rating vs Views)
sns.scatterplot(x="Views (Million)", y="Avg Rating", data=df, ax=axes[1], hue="Category", s=100)
axes[1].set_title("Movie Ratings vs Views")
axes[1].set_ylabel("Average Rating")

# Final touches
plt.tight_layout()
plt.show()

# Insights
most_popular = df.loc[df["Views (Million)"].idxmax()]
print(f"🔥 Most Watched Category: {most_popular['Category']} with {most_popular['Views (Million)']}M views")
print(f"⭐ Average Rating Trend: High views do not always mean high ratings!")
