import numpy as np

X=[8.33333333e+00, 4.11522634e-03, 1.81694583e+01, 8.65076540e-02]
top_50=np.percentile(X,75)
indices = np.where(X <= top_50)[0]

print(top_50,indices)
