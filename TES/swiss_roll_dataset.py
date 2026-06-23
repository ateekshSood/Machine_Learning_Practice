# %%

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


data , target = make_moons(n_samples = 1000 , noise = 0.15 ,  random_state = 42 )

x_train , x_test , y_train , y_test = train_test_split(data , target , random_state = 42)


# %%

import matplotlib.pyplot as plt 

plt.scatter(x_train[: , 0]  , x_train[: , 1], c=y_train)
plt.axis('off')





plt.show()
# %%

from sklearn.datasets import make_swiss_roll 

data , target = make_swiss_roll(n_samples = 1000 , noise = 0.15 , random_state = 42)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$" , labelpad=-1)
plt.tight_layout()



scatter = ax.scatter(data[: , 0] , data[: , 1] , data[: , 2] , c=target , s= 30)
plt.show()


# %%

plt.scatter(data[: , 0] , data[: , 1] , c=target , s=30)
plt.axis('off')
plt.show()
# %%

import seaborn as sns

sns.scatterplot(x=data[: , 0] , y=data[: , 1] , hue=target , palette = 'viridis' , legend=False)
plt.axis('off')
plt.show()
# %%

#normla pca


from sklearn.decomposition import PCA 

fig = plt.figure(figsize=(12 , 5))


ax0 = fig.add_subplot(121)
pca = PCA(n_components = 2)
data_pca = pca.fit_transform(data)
ax0.scatter(
    data_pca[: , 0],
    data_pca[: , 1],
    c=target,

)

ax0.set_title("2D PCA")
plt.axis('off')

ax1 = fig.add_subplot(122, projection='3d')

ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$x_3$" , labelpad=-1)
scatter = ax1.scatter(data[: , 0] , data[: , 1] , data[: , 2] , c=target , s= 30)
ax1.set_title("Original Dataset")
plt.axis('off')

plt.show()
