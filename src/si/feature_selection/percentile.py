from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class percentile:
    def __init__(self, score_func: Callable = f_classification, percentile:int=50 ) -> None:
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit (self,dataset:Dataset) -> "percentile":
        self.F, self.p = self.score_func(dataset)
        return self
        pass

    def transform (self, dataset:Dataset) -> Dataset:
        top_50=np.percentile(self.F,self.percentile)
        features=np.array(dataset.features)[top_50]
        return Dataset(X=dataset.X[:,top_50], y=dataset.y, features=features, label=dataset.label) 
    
    def fit_transform(self,dataset:Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)