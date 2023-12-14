import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

from typing import Tuple, Union

import numpy as np
from scipy import stats

from src.si.data.dataset import Dataset

def f_classification(dataset:Dataset) -> Union[Tuple[np.ndarray,np.ndarray], Tuple[float,float]]:
    """
    provides the one-way Anova for the proivided dataset, and return the F and P value.
    The F-value scores allows analyzing if the mean between two or more groups (factors) are significantly different. 
    Samples are grouped by the labels of the dataset.

    Parameters
    -------
    dataset: Dataset
        A labeled dataset

    Returns
    -------
    F: np.array, shape (n_features,)
        F scores
    p: np.array, shape (n_features,)
        p-values
    """

    classes=dataset.get_classes() # vai me dar as classes unicas existentes no dataset (possiveis valores de label y)- função ja criada
    groups=[dataset.X[dataset.y==c,:] for c in classes] # associar cada samples a uma determinado classe e assim obtenho os meus grupos
    F,p=stats.f_oneway(*groups) #fazer a anova
    return F,p


#testing
if __name__ == '__main__':
    from src.si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 5],
                                  [0, 1, 1, 2]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")



    # Chamar a função f_classification para calcular F-score e p-score
    F_score, p_score = f_classification(dataset)

    # Imprimir os resultados
    print("F-score:", F_score)
    print("p-score:", p_score)
