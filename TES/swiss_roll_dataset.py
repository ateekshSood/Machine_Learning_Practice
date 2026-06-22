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

scatter = ax.scatter(data[: , 0] , data[: , 1] , data[: , 2] , c=target , s= 30)
