import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np

def manhattan_distance(x:np.ndarray,y:np.ndarray)->np.ndarray:
    """
    this fuction allow to calculate the manhattan distance of a point (X) to a set of point y.
    distance_x_y1 = |x1 - y11| + |x2 - y12| + ... + |xn - y1n|
    distance_x_y2
    etc

    Parameters
    ------
    x:point -point that we pretende to now the distance
    y:set of points -different points that is differente clusters/centroids and off course that in the end we want the lower distance

    Returns
    -----
    the manhattan distance for each point iin y 
    """
    return np.abs((x-y).sum(axis=1))#eixo 1 porque vamos obter uma matriz de 1 linhas por muitas colunas , mas apenas queremos somar as linhas

#testar
x = np.array([3, 4])

y = np.array([[1, 2],
              [5, 6],
              [0, 0]])

our_distance = manhattan_distance(x, y)
# using sklearn
# to test this snippet, you need to install sklearn (pip install -U scikit-learn)
from sklearn.metrics.pairwise import manhattan_distances
sklearn_distance = manhattan_distances(x.reshape(1, -1), y)
assert np.allclose(our_distance, sklearn_distance)
print(our_distance, sklearn_distance)






