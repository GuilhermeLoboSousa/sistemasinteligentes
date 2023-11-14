import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
import random

output_error = np.array([[0.8, 0.3, 0.1]])
a=np.sum(output_error, axis=0, keepdims=True)
print(a)