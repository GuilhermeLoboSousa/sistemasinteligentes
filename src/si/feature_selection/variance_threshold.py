
import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

import numpy as np
from src.si.data.dataset import Dataset

class VarianceThreshold:
    """
    The goal is to do a feature seletion (filter) by a certain variance threshold.
    This is, featuers with a variance higher than this threshold will be removed from the dataset.

    Parameters
    -------------
    threshold-florat
        value used for the feature selection
        Features with a training-set variance lower than this threshold will be removed.

    Estimated parameters
    ------------------
    Variance: array-like, shape (n_features,)
        the varaince of each featue estimated from data
    """
    def __init__(self, threshold:float=0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.

        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("the threshold must be a positive value")
        
        self.threshold=threshold #parameters

        self.variance=None #estimated parameters
    
    def fit (self,dataset:Dataset) -> "VarianceThreshold":
        
        """
        Fit the VaraianceThresold model according to the given data, basicly estimates the variance of each feature.
        This method is responsible for estimating parameters from the data (variance in this case)
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.

        Returns
        -------
        self : object
        """
        self.variance=dataset.get_variance() #metodo criado na outra classe- diferente do prof
        return self 
    
    def transform(self,dataset:Dataset) -> Dataset:
        """
        this method is responsible for transforming de data
        would remove all features whose variances does not meet the treshold
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        X=dataset.X
        features_rule=self.variance > self.threshold #vai dar true ou false para ssaber onde a condição é verdade
        X=X[:,features_rule] #escolho todas as linhas e as colunas onde apenas tive true (filtro)
        features=np.array(dataset.features)[features_rule]#apenas quero guardas as features que foram selecionadas apos aplicar o filtro
        return Dataset(X,y=dataset.y,features=list(features),label=dataset.label)
    
    def fit_transform(self,dataset:Dataset) -> Dataset:
        """
        Runs fit to data and the transform it
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        self.fit(dataset)
        return self.transform(dataset)
    

#testing
if __name__ == '__main__':
    from src.si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)

    thresholds = [0.1, 0.5, 1.0,-0.4]
    for threshold in thresholds:
        selector = VarianceThreshold(threshold=threshold)
        selector = selector.fit(dataset)
        dataset_filtered = selector.transform(dataset)
        print(f"Features for threshold {threshold}: {dataset_filtered.features}")



