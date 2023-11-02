import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
import random

parameter_grid_ = {
    'l2_penalty': (1, 10),
    'alpha': (0.001, 0.0001),
    'max_iter': (1000, 2000)
}

model="knn"
for b in range(2):
    dic={}
    for x,y in parameter_grid_.items():
        a=np.random.choice(y)
        dic[x]=a
        c=setattr(model,x,a)
        print(c)
    print(dic)