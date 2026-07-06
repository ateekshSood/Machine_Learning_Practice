# %%
import pandas as pd
import os
# %%

df = pd.read_csv("machine_learning_practice/rev/datasets/AmesHousing.csv")

# %%

df.info()
# %%



df.describe()
# %%
import matplotlib.pyplot as plt
df.hist(bins = 50 , figsize = (20, 20))
plt.show()
# %%
print(df["Garage Yr Blt"].max())
print(df["Year Built"].max())


# %%

print(df["Garage Yr Blt"].isna().sum())


# %%

print(df[df["Garage Yr Blt"].isna()]["Garage Cars"].unique())
oddGarageBlt = df[(df["Garage Yr Blt"].isna()) & (df["Garage Cars"] == 1)]
print(oddGarageBlt["Garage Area"])
print(oddGarageBlt["Garage Type"])
# %%

import numpy as np

df["has_garage"] = (df["Garage Cars"] > 0).astype(int)
garage_yr_filled = df["Garage Yr Blt"].fillna(df["Year Built"])
df["garage_age"] = df["Yr Sold"] - garage_yr_filled
df["garage_age"] = np.where(df["has_garage"] == 1 , df["garage_age"] , 0)
df = df.drop("Garage Yr Blt" , axis= 1 )

# %%

df.groupby("has_garage")["garage_age"].describe()
