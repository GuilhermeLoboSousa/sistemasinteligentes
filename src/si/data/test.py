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

# Número de classes
num_classes = np.max(y) + 1

# Inicializa matrizes para armazenar a contagem de 0s e 1s por classe e coluna
count_zeros_by_class = np.zeros((num_classes, X.shape[1]), dtype=int)
count_ones_by_class = np.zeros((num_classes, X.shape[1]), dtype=int)

# Loop sobre cada classe
for class_label in range(num_classes):
    # Obtém os índices das amostras pertencentes a esta classe
    class_indices = np.where(y == class_label)
    
    # Seleciona as amostras da classe atual
    class_samples = X[class_indices]
    
    # Calcula a contagem de 0s e 1s em cada coluna
    count_zeros = np.sum(class_samples == 0, axis=0)
    count_ones = np.sum(class_samples == 1, axis=0)
    
    # Armazena as contagens nas matrizes apropriadas
    count_zeros_by_class[class_label] = count_zeros
    count_ones_by_class[class_label] = count_ones

# Imprime as contagens por classe
for class_label in range(num_classes):
    print(f"Classe {class_label} - Contagem de 0s por coluna: {count_zeros_by_class[class_label]}")
    print(f"Classe {class_label} - Contagem de 1s por coluna: {count_ones_by_class[class_label]}")

import numpy as np


# Número de classes
num_classes = np.max(y) + 1

# Inicializa matrizes como float para armazenar a contagem de 0s e 1s por classe e coluna
count_zeros_by_class = np.zeros((num_classes, X.shape[1]), dtype=float)
count_ones_by_class = np.zeros((num_classes, X.shape[1]), dtype=float)

# Loop sobre cada classe
for class_label in range(num_classes):
    # Obtém os índices das amostras pertencentes a esta classe
    class_indices = np.where(y == class_label)
    
    
    # Seleciona as amostras da classe atual
    class_samples = X[class_indices]
    
    # Divide a contagem de 0s e 1s em cada coluna pelo número de amostras da classe
    count_zeros = np.sum(class_samples == 0, axis=0)
    count_ones = np.sum(class_samples == 1, axis=0)
    
    count_zeros_by_class[class_label] = count_zeros / len(class_indices[0])
    count_ones_by_class[class_label] = count_ones / len(class_indices[0])

# Imprime as listas divididas
for class_label in range(num_classes):
    print(f"Classe {class_label} - Contagem de 0s por coluna: {count_zeros_by_class[class_label]}")
    print(f"Classe {class_label} - Contagem de 1s por coluna: {count_ones_by_class[class_label]}")





  