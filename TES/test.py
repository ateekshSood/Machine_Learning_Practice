import numpy as np

# BOOLEAN STUFF 

# Imagine you have a list of numbers representing the "Categories" of items in a small dataset:
# categories = [0, 1, 2, 0, 3, 2, 1, 0]

# Question 1: The "Bouncer" Test
# Goal: Create a list of True and False (a mask) that identifies only the zeros.

categories = np.array([0, 1, 2, 0, 3, 2, 1, 0])
mask = categories == 0
print(mask)

# Question 2: The "Either/Or" (The | Operator)
# Goal: You want to find both category 0 and category 2.

