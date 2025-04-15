import pandas as pd
import numpy as np

df = pd.read_csv("Book1.csv")
df.columns = df.columns.str.strip()

sort_func = df.sort_values(by=['Item'])
print("Sorted value is: \n", sort_func)

filter_row = df.query('Rate > 5000')
print("Filter by row (Rate > 5000): \n", filter_row)

filter_column = df.filter(['Item', 'Rate'])
print("Filter by column (Item & Rate only): \n", filter_column)

group_data = df.groupby('Item')
print("Grouped data (First value of each group): \n", group_data.first())
