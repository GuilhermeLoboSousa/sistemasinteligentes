import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

from typing import Callable

import numpy as np
import warnings

from src.si.data.dataset import Dataset
from src.si.statistics.f_classification import f_classification

class Percentile:
    """
    Select a certain percentage of the features taking into account the F-score value.
    this is first we see the f-score of each feature and sorted that.
    after we choose a percentil that representes x % of this f-values sorted
    so we keep the features that indices have the f-value <= to the percentile
    
    Parameters
    -----------
    score_func:callable 
        taking the dataset and return a pair os array (F and p value)- allow analize the variance 
    percentile: int, deafult 50
        number that represents a percentage of the data/features to select 

    estimated parameters(given by the score_func)
    ---------------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, percentile:int=50 ) -> None:
        """
        Select features according to the percentile.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        percentile: int, default=50
            Percentile of top features to select.
        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

        if self.percentile > 100 or self.percentile < 0:
            raise ValueError("the value of percentile must be between 0 and 100")

    def fit (self,dataset:Dataset) -> "percentile":
        """
        Estimates the F and p values for each feature using the score_func

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        if np.isnan(dataset.X).any():
            warnings.warn("Caution: The dataset contains NaN values which can lead to incorrect results when computing statistics. You may want to use some preprocessing methods like dropna or fillna.")        
        self.F, self.p = self.score_func(dataset)
        self.F = np.nan_to_num(self.F)

        return self

    def transform (self, dataset:Dataset) -> Dataset:
        """
        select the percentage of the data (percentile) that we want taking into acount the F-value of each feature

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the highest scoring features within a specified percentile.
        """
        bottom_50=np.percentile(self.F,self.percentile) #nao preciso de ordenar primeiro os valores, para calcular o percentile a ordem nao importa(ja testei isso)
        indices = np.where(self.F > bottom_50)[0] # indices vai dizer a posição dos valores de F que têm valor maior que o percentile, ou seja identifica os indices das colunas que devem ser escolhidas
        features=np.array(dataset.features)[self.F > bottom_50] #mesma logica mas para ir buscar o nome das mesmas
        return Dataset(X=dataset.X[:,indices], y=dataset.y, features=features, label=dataset.label) #dataseet.X aparece sem ordem particular, ou seja apenas apresenta as features que estao acima desse percentil
    
    def fit_transform(self,dataset:Dataset) -> Dataset:
        """
        Runs fit and then transform the dataaset taking into acount a certain percentile of the fvalue score.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the highest scoring features within a specified percentile.
        """        
        self.fit(dataset)
        return self.transform(dataset)
    


#testing
if __name__ == '__main__':
    from src.si.data.dataset import Dataset
    from sklearn.feature_selection import SelectPercentile, f_classif

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                               [0, 1, 4, 3],
                               [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    select_percentile = Percentile(percentile=50)
    select_percentile.fit(dataset)
    dataset = select_percentile.transform(dataset)
    print(dataset.features)


    #percentiles = [50,25,75]
    #for percentile in percentiles:
        #selector = Percentile(percentile=percentile)
        #selector = selector.fit(dataset)
        #dataset_filtered = selector.transform(dataset)
        #print(f"Features for percentile {percentile}: {dataset_filtered.features}")
        #print(dataset_filtered.X)

 