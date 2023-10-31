import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np

def entropy_impurity(y:np.ndarray)->float: #quando maior for a entropia pior foi o "split" realizado
    """
    Calculates the impurity of a dataset using entropy.
    Total entropy, that is, entropy for each "class" that was chosen

    Parameters
    ----------
    y: np.ndarray
        The labels of the dataset.

    Returns
    -------
    float
        The impurity of the dataset.
    """
    labels,counts=np.unique(y,return_counts=True)
    impurity=0 #devido ao sinal - que tem na formula apresentada no slide
    for x in range(len(labels)):#tem de ser feito para todas as classes ; um aparte ao fazer isto dá caso a label seja uma letra ou numero , se fizesse for x in labels já so dava se labels fosse numero
        impurity -= (counts[x]/len(y))*np.log2(counts[x]/len(y))
    return impurity

def gini_impurity(y:np.ndarray)->float: #quando menor for gini melhor foi o "split" realizado
    """
    Calculates the impurity of a dataset using the Gini index.

    Parameters
    ----------
    y: np.ndarray
        The labels of the dataset.

    Returns
    -------
    float
        The impurity of the dataset.
    """
    labels,count=np.unique(y,return_counts=True)
    impurity=1
    for x in range(len(labels)):
        impurity -= (count[x]/len(y))**2
    return impurity