import pandas as pd
import json
df = pd.read_csv("Book1.csv")
print(df)

student_json = [
    {"name":"alicie"},
    {"name":"dddd"},
    {"name":"dd"}
]

df_J = pd.DataFrame(student_json)
print(df_J)