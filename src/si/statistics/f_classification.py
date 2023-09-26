from typing import Tuple, Union

import numpy as np
from scipy import stats

from si.data.dataset import Dataset

def f_classification(dataset:Dataset) -> Union[Tuple[np.ndarray,np.ndarray], Tuple[float,float]]:
    """
    provides the one-way Anova for the proivided dataset, and return the F and P value.
    The F-value scores allows analyzing if the mean between two or more groups (factors) are significantly different. 
    Samples are grouped by the labels of the dataset.

    Parameters
    -------
    dataset

    return
    -----
    F-score
    p-score
    """

    classes=dataset.get_classes() # vai me dar as classes unicas existentes no dataset (possiveis valores de label y)- função ja criada
    groups=[dataset.X[dataset.y==c,:] for c in classes] # associar cada samples a uma determinado classe e assim obtenho os meus grupos
    F,p=stats.f_oneway(*groups) #fazer a anova
    return F,p
