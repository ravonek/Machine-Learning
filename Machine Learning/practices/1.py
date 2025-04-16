import pandas as pd
import numpy as np


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
data = pd.read_csv(url, delim_whitespace=True, header=None)

data.head()

df = data.copy()
df.columns = ["mpg", "cylinders",
"displacement", "horsepower", "weight","acceleration", "model_year", "origin",
"car_name"]

df["horsepower"] = pd.to_numeric(df["horsepower"], errors='coerce')

df = df.dropna()

# print(df.info())



df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna()

def flag_bestsellers(d):
    if (d['cylinders'] == 8 and d['model_year'] == 70):
        return 1
    elif(d['cylinders'] == 6 and d['model_year'] == 80):
        return 2
    else:
        return 0

df['bestsellers'] = df.apply(flag_bestsellers, axis = 1)
df.head()


# print(df.dtypes)

avg_hp = df['horsepower'].mean()
min_w = df['weight'].min()

max_mpg = df.groupby(['model_year', 'cylinders'])['mpg'].max().sort_values(ascending=False)


filtered = df[(df['weight'] < 3449) & (df['cylinders'] > 5)]
print(f"Количество машин с весом < 3449 и цилиндрами > 5: {filtered.shape[0]}")

print(max_mpg.max())
print(avg_hp)
print(min_w)

df.info()
df.describe()





