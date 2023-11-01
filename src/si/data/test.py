import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
predictions = np.zeros((2, 10), dtype=object)#matriz onde as linhas serao as arvores realizadas e as colunas a previsao destas arvores para cada sample
v=["a","b","c","a","b","c","a","b","c","a"]
predictions[0,:]=v
x=['Iris-versicolor', 'Iris-setosa' ,'Iris-virginica', 'Iris-versicolor',
 'Iris-versicolor', 'Iris-setosa' ,'Iris-versicolor' ,'Iris-virginica',
 'Iris-virginica' ,'Iris-versicolor' ,'Iris-virginica', 'Iris-setosa',
 'Iris-setosa']
b=Counter(x)
print(b.most_common(1)[0])