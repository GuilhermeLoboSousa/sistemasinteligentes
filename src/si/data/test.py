import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, np.nan],
              [7, np.nan, 9]])
y = np.array(["a", "b", "a"])
features = ['feature_1', 'feature_2', 'feature_3']
label = 'target'

#print(np.random.permutation(X.shape[0]))

labels, counts = np.unique(y, return_counts=True)
print(labels,counts)

#for x in labels:
    #print(int(counts[x]*0.4))
