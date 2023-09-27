import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

from typing import Callable

import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectKBest:
    """
    select features according to the k highest score(some statistic function)
    Feature ranking is performed by computing the scores of each feature using a scoring function(statistic):
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks another statis test not implemented yet).
    
    Parameters
    -----------
    score_func: taking the dataset and return a pair os array (F and p value)- allow analize the variance 
    k: nuber of top features that we want

    estimated parameters(given by the score_func)
    ---------------
    F:
    p:

    """

    def __init__(self, score_func:Callable= f_classification, k:int =10) : #callable permite invocar uma função como objeto i guess
        """
        same logic that before
        
        """
        self.K=k
        self.score_func=score_func
        self.F=None
        self.p=None
    
    def fit (self, dataset:Dataset) -> "SelectKBest":
        """
        estimates the F and p values for each feature using the score_func

        Parameters
        -------
        Dataset

        Return---
        itself
        """

        self.F,self.P=self.score_func(dataset) #vai chamar a função f_classification ja criada que da como return o tupple F e p
        return self
    
    def transform(self,dataset:Dataset) ->Dataset:
        """
        select the top k features accord the F-value

        Returns:
        the select (filtered) dataset
        """
        top_10=np.argsort(self.F)[-self.K:] # ordena os valores de F de forma CRESCENTE e seleciona os 10 ultimos neste caso, que sao os top_10
        features=np.array(dataset.features)[top_10] #apenas fico com as features do top 10
        return Dataset(X=dataset.X[:,top_10], y=dataset.y, features=features, label=dataset.label) #faço aqui a filtragem do dataset X-todas as linhas , mas apenas as 10 melhores colunas
    
    def fit_transform(self,dataset:Dataset): #basicamente este junta as duas funções
        """
        Runs fit and thaen transform the dataaset by electing the k highest scoring features.

        return
        ------
        dataset already filtered with the k highest scoring features
        """
        self.fit(dataset)
        return self.transform(dataset)
    

#testing
if __name__ == '__main__':
    from src.si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0.2, 2, 0.06, 2.87],
                                  [0.5, 1.5, 4, 3],
                                  [0.3, 1, 1, 3.4]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")


    ks = [1,2,3]
    for k in ks:
        selector = SelectKBest(k=k)
        selector = selector.fit(dataset)
        dataset_filtered = selector.transform(dataset)
        print(f"Features for k {k}: {dataset_filtered.features}")




