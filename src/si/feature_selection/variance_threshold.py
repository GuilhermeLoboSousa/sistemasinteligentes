import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub")
import numpy as np
from src.si.data.dataset import Dataset

class VarianceThreshold:
    """
    The goal is to do a feature seletion (filter) by a certain variance threshold.
    This is, featuers with a variance higher than this threshold will be removed from the dataset.

    Parameters
    -------------
    threshold-value used for the feature selection

    Estimated parameters
    ------------------
    Variance: the varaince of each featue estimated from data
    """
    def __init__(self, threslhold:float=0.0):
        """
        same logic explain before
        """
        if threslhold < 0:
            raise ValueError("the threshold must be a positive value")
        
        self.threshold=threslhold #parameters

        self.variance=None #estimated parameters
    
    def fit (self,dataset:Dataset) -> "VarianceThreshold":
        
        """
        Fit the VaraianceThresold model according to the given data, basicly estimates the variance of each feature.
        This method is responsible for estimating parameters from the data (variance in this case)
        Return itself
        """
        self.variance=dataset.get_variance() #metodo criado na outra classe
        return self #aqui nao percebi bem
    
    def transform(self,dataset:Dataset) -> Dataset:
        """
        this method is responsible for transforming de data
        would remove all features whose variances does not meet the treshold
        """
        X=dataset.X
        features_rule=self.variance > self.threshold #vai dar true ou false para ssaber onde a condição é verdade
        X=X[:,features_rule] #escolho todas as linhas e as colunas onde apenas tive true (filtro)
        features=np.array(dataset.features)[features_rule]#apenas quero guardas as features que foram selecionadas apos aplicar o filtro
        return Dataset(X,y=dataset.y,features=list(features),label=dataset.label)
    
    def fit_transform(self,dataset:Dataset) -> Dataset:
        """
        runs fit to data and the transform it
        """
        self.fit(dataset)
        return self.transform(dataset)
    

#testing
if __name__ == '__main__':
    from si.data.dataset import Dataset

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



