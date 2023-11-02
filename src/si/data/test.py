import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
a=[]
b=[1,0,1]
c=[1,1,1]
d=[0,0,1]

stacked_data = np.column_stack([b, d, d])
print(stacked_data)
