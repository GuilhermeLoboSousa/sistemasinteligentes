import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, np.nan],
              [7, np.nan, 9]])
y = np.array([0, 1, 0])
features = ['feature_1', 'feature_2', 'feature_3']
label = 'target'

print(np.random.permutation(X.shape[0]))