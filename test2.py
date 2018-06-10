import numpy as np

X = np.array([ [ 1,2,3],
      [4,5,6]])

Y = np.array([[ 7,8,9]])




permutation = list(np.random.permutation(3))
shuffled_X = X[:, permutation]
shuffled_Y = Y[:, permutation].reshape((1, 3))


print(shuffled_X)
print(shuffled_Y)