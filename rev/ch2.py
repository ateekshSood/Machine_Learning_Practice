# %%

#the housing dataset  

from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()


# %%

housing_full.head()

# %%
housing_full.info()
# %%

housing_full["ocean_proximity"].value_counts()
housing_full.describe()
# %%

import matplotlib.pyplot as plt

housing_without_category = housing_full.drop(columns = "ocean_proximity")

housing_without_category_col_list = housing_without_category.columns.tolist() 
housing_without_category_col_list

# %%

# from itertools import product
# fig , ax = plt.subplots(3 , 3 )

# for (iter , (x , y)) in enumerate(product(range(3) , range(3))):

#     ax[x , y].hist(housing_without_category[housing_without_category_col_list[iter]])
#     ax[x,y].grid()
#     ax[x,y].set_title(housing_without_category_col_list[iter])
housing_full.hist(bins=50 , figsize=(12 , 8))
plt.show()

# %%
from sklearn.model_selection import train_test_split 


train_set , test_set = train_test_split(housing_full , test_size = 0.2 , random_state = 42)
 
# %%
import numpy as np

housing_full["income_cat"] = pd.cut(housing_full["median_income"] , 
    bins = [0. , 1.5 , 3 , 4.5 , 6. , np.inf ],
    labels = [1 , 2  , 3 ,4  ,5 ])



# %%

housing_full["income_cat"].hist()
plt.show()
# %%

cat_graph = housing_full["income_cat"].value_counts().sort_index()
cat_graph.plot.bar(rot = 0 , grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
# %%

from sklearn.model_selection import StratifiedShuffleSplit 

strat_train_set , strat_test_set =  train_test_split(
    housing_full , test_size = 0.2 , stratify = housing_full["income_cat"] , random_state = 42
)
# %%

strat_train_set["income_cat"].value_counts().sort_index().plot.bar(grid=True , rot=0)

# %%
strat_train_set.drop("income_cat" , axis=1 , inplace=True)
strat_test_set.drop("income_cat" , axis=1 , inplace=True)

# %%

housing = strat_train_set.copy()

# %%

import seaborn as sns

sns.scatterplot(data=housing , x = "longitude" , y = "latitude" , hue="median_house_value" , palette="magma")
plt.grid()

# plt.scatter(housing["longitude"] , housing["latitude"] ,  s=10) 

plt.show()
# %%

corr_matrix = housing.corr(numeric_only = True )
corr_matrix["median_income"].sort_values(ascending=False)
# %%

from pandas.plotting import scatter_matrix 

attributes = ["median_income" , "median_house_value" , "total_rooms" , "housing_median_age"]

scatter_matrix(housing[attributes] , figsize=(12 , 8))
plt.show()
# %%

housing["rooms_per_house"] = housing["total_rooms"]/housing["households"]
housing["bedroom_ratio"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["people_per_house"] = housing["population"]/housing["households"]
# %%

corr_matrix = housing.corr(numeric_only = True)
corr_matrix["median_house_value"].sort_values(ascending=False)
# %%

housing = strat_train_set.drop("median_house_value" , axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# %%

housing.info()
# %%

housing["total_bedrooms"].isnull().sum()

# %%

housing_total_bedromms_median = housing["total_bedrooms"].median()
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(housing_total_bedromms_median)
# %%

