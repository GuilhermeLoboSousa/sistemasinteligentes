import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
import random

input = np.array([[0.8, 0.3,-0.7],[0.1, 0.9,0]])

a=input - np.max(input, axis=0, keepdims=True)
print(a)