import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn
import numpy as np

import numpy as np

predictions = [1, 2, 0.4, -0.5]
predictions = np.array(predictions)  # Converter a lista em uma matriz NumPy
a = np.where(predictions >= 0.5, 1, 0)
print(a)

mask = predictions >= 0.5
predictions[mask] = 1
predictions[~mask] = 0
print(predictions)


  