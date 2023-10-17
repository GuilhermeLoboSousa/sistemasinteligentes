import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
import numpy as np

# Seu conjunto de dados (X representa as características binárias e y as classes)
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 0, 0],
              [1, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [1, 1, 1]])
y = np.array([0,0,1,1,0,0,1,1,1,0])


  