import pandas as pd
import numpy as np

df = pd.read_csv("car.csv")
df.columns = df.columns.str.strip()


filter_row = df.query('BuyPrice > 5000')
print("Filter by row (Buy Price > 5000): \n", filter_row)

sort_func = df.sort_values(by=['Model'])
print("Sorted value is: \n", sort_func)



# filter_column = df.filter(['Item', 'Rate'])
# print("Filter by column (Item & Rate only): \n", filter_column)

group_data = df.groupby('Model')
print("Grouped data (First value of each group): \n", group_data.first())
