import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np

def euclidean_distance(x:np.ndarray,y:np.ndarray)->np.ndarray:
    """
    this fuction allow to calculate the eucledian distance of a point (X) to a set of point y.
    so is to apply the pitagoras theorem 
    distance_y1n = sqrt((x1 - y11)^2 + (x2 - y12)^2 + ... + (xn - y1n)^2)
    distance_y2n = sqrt((x1 - y21)^2 + (x2 - y22)^2 + ... + (xn - y2n)^2
    etc

    Parameters
    ------
    x:point -point that we pretende to now the distance
    y:set of points -different points that is differente clusters/centroids and off course that in the end we want the lower distance

    Returns
    -----
    the eucledian distance for each point iin y 
    """
    return np.sqrt(((x-y)**2).sum(axis=1))#eixo 1 porque vamos obter uma matriz de 1 linhas por muitas colunas , mas apenas queremos somar as linhas
