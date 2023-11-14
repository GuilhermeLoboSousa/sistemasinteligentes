import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
import random

input = np.array([[0.8, 0.3,0],[0.1, 0.9,0]])
mask = np.random.binomial(1, 1 - 0.5, size=input.shape)

print(input.shape[0])