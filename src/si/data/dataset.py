import numpy as np
import pandas as pd
from typing import Tuple, Sequence, Union

class Dataset:
    """
    Class representing a tabular dataset for single output classification.
    """
    def __init__ (self, X:np.ndarray ,y:np.ndarray, features:list[str]=None, label:str=None) -> None:
        """
        Dataset is a structured collection of data used for training, validating, and testing machine learning models.

        Attributes
        -------------
        X: numpy.ndarray (n_samples,n_features)
            matrix of features (the values that characterized the features)- like is a matrix i use ndarray and not only array.(dimension would be the number of samples/subjects by the number of features)
        y: numpy.ndarray
            vector so dimensions would be n_samples/n_subjects,1
        features: list of str (n_features) 
            vector of feature names
        label: str
            name of the dependent variable (only one)
        if logic:
        1-must exist the matrix X is the most important rule
        2-if X exist and y do not, so y assum the str y_empty
        3-if we have y , so this y must have the same number of lines that X, this is the same number of samples/subjects
        4-we must have the same number of columns and features
        5- if we dont have name for the features we assume feat plus number of this column
        6-we dont want features with the same name , so at least one letter or number must be different
        """
        if X is None:
            raise ValueError ("must exist a matrix X")
        if X is not None and label is None:
            label="y_empty"
        if y is not None and len(X) != len(y):
            raise ValueError("x e y must have the same number of lines")
        if features is not None and X.shape[1] != len(features):
            raise ValueError("must have the same number of features and columns")
        if features is None:
            features=[]
            for x in range(X.shape[1]):
                features.append(f"feat_{str(x)}")
        if features is not None:
            unique_features = set()
            for feature_name in features:
                if feature_name in unique_features:
                    raise ValueError("Duplicated feature name")
                unique_features.add(feature_name)

        self.X=X
        self.y=y
        self.features=features
        self.label=label

    def shape (self) -> tuple[int,int]:
        """
        The return would be  a tupple with the numer of lines/samples/subjects and collumns/features
        
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape
    
    def has_label (self) -> bool: # we want to see if exist a label y
        """
        Returns True if the dataset has a label

        Returns
        -------
        bool
        """
        return False if self.y is None else True
    
    def get_classes (self) -> np.ndarray:
        """
        Return the classes of the dataset, this is, the possivel values of label y
        
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label() is True:
            return np.unique(self.y)
        else:
            raise ValueError("we dont have the label y")
        
    def get_mean(self) -> np.ndarray:
        """
        Return the mean of each collumn /feature

        Returns
        -------
        numpy.ndarray (n_features) 
        """
        return np.nanmean(self.X, axis=0) #o eixo que estamos a prender
    
    def get_variance(self) -> np.ndarray:
        """
        Return the variance of each collumn/feature

        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X,axis=0) #nan tem em conta já os valores nao definidos
    
    def get_median(self) -> np.ndarray:
        """
        Return the median of each collumn/feature

        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X,axis=0) 

    def get_max(self) -> np.ndarray:
        """
        Return the maximum value of each collumn/feature

        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X,axis=0) 

    def get_min(self) -> np.ndarray:
        """
        Return the minimum value of each collumn/feature

        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X,axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Return a summary of the dataset with the functions create before

        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data={"mean": self.get_mean(),
                "variance": self.get_variance(),
                "median": self.get_median(),
                "max": self.get_max(),
                "min":self.get_min()
                }

        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)
    
    def dropna(self) -> np.ndarray:
        """
        This fuction allow to remove the samples that have nan values and behind that give return self and the index of this lines
        
        Returns
        -------
        tuple (self, np.ndarray(indice))
        """
        without_na=np.isnan(self.X).any(axis=1)
        self.X=self.X[~without_na]
        self.y=self.y[~without_na]
        indice=np.where(without_na)
        return self,indice #acrecentei para conseguirmos saber que samples tinham NA values
    
    def fill_na(self, strategy:str=None):
        """
        This function allow replace de null values by the :
        first option: a value random choose betwween the min and max value of the column with nan values
        second: the median value of the column
        third: the mean value of the collumn
        Arguments
        -------
        strategy: str
            We can choose if want to replace by a number randomly choosen betwwen min-max, median or mean
        """
        if strategy is None:
            raise ValueError("please, put the value or mean or median")
        columns_true=np.isnan(self.X).any(axis=0) #ter bool onde true significa que tem um valor nulo, dá lista de colunas com true ou false consoante tem ou nao valores nulos
        nan_columns_indices = np.where(columns_true)[0] # vou buscar os indices onde existem os bool a true
        

        for col_index in nan_columns_indices:
            col = self.X[:, col_index]#guardar as colunas que tem valores nulos, ou seja vai buscar a coluna com valores nulos
            
            if strategy == "value": #opto por escolher entro o maximo e o minimo
                min_value = np.nanmin(col)#perguntar ao prof se posso usar a função get minimu
                max_value = np.nanmax(col)
                final = np.random.uniform(min_value, max_value) #algo aleatorio entre o minimo e o maximo
            elif strategy == "median":
                final = np.nanmedian(col)
            elif strategy == "mean":
                final = np.nanmean(col)
            
            col[np.isnan(col)] = final # vou buscar os valores nan como true(dentro da coluna já identificada como ter esses valores) e depois , é basicamente col[onde é true?] e substituir pelo que quero

        return self
    

    def remove_by_index (self,index:int=None):
        """
        Remove a line from the dataset by the given index.

        Arguments:
        ----------
        index: int
            The index of the line to be removed.

        Returns
        -------
        self
        """
        if not isinstance(index, int):
            raise ValueError("Please provide a valid integer index.")
        
        if index <0 and index> self.X.shape[0]:
            raise ValueError("Put a valid index")
        
        self.X = np.delete(self.X, index, axis=0) #remover algo do eixo linhas do dataset X e esse algo é a posição index
        if self.y is not None:#caso exista y vou quere apagar tb o seu index
            self.y = np.delete(self.y, index)
        return self

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)

if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print(dataset.shape())
    print(dataset.has_label())
    print(dataset.get_classes())
    print(dataset.get_mean())
    print(dataset.get_variance())
    print(dataset.get_median())
    print(dataset.get_min())
    print(dataset.get_max())
    print(dataset.summary())