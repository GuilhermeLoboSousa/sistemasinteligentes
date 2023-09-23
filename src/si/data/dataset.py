import numpy as np
import pandas as pd
class Dataset:
    def __init__ (self, X:np.ndarray ,y:np.ndarray, features:list[str]=None, label:str=None) -> None:
        """
        Dataset is a structured collection of data used for training, validating, and testing machine learning models.

        Attributes
        -------------
        X- matrix of features (the values that characterized the features)- like is a matrix i use ndarray and not only array.(dimension would be the number of samples/subjects by the number of features)
        y-vector so i have doubts, because maybe can only use np.array (dimensions would be n_samples/n_subjects,1)
        features- vector of feature names
        label- name of the dependent variable (only one)
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
            the result would be  a tupple with the numer of lines/samples/subjects and collumns/features
            """
            return self.X.shape
        
        def has_label (self) -> bool:
            """
            we want to see if exist a label y
            """
            return False if self.y is None else True
        
        def get_classes (self) -> np.ndarray:
            """
            return the class of the dataset, this is, the possivel values of label y
            """
            if self.has_label() is True:
                return np.unique(self.y)
            else:
                raise ValueError("we dont have the label y")
            
        def get_mean(self) -> np.ndarray:
            """
            return the mean of each collumn /feature 
            """
            return np.nanmean(self.X, axis=0) #o eixo que estamos a prender
        
        def get_variance(self) -> np.ndarray:
            """
            return the variance of each collumn/feature
            """
            return np.nanvar(self.X,axis=0) #nan tem em conta já os valores nao definidos
        
        def get_median(self) -> np.ndarray:
            """
            return the median of each collumn/feature
            """
            return np.nanmedian(self.X,axis=0) 

        def get_max(self) -> np.ndarray:
            """
            return the maximum value of each collumn/feature
            """
            return np.nanmax(self.X,axis=0) 

        def get_min(self) -> np.ndarray:
            """
            return the minimum value of each collumn/feature
            """
            return np.nanmin(self.X,axis=0)

        def summary(self) -> pd.Dataframe:
            """
            return a summary of the dataset with the functions create before
            """
            data={"mean": self.get_mean(),
                  "variance": self.get_variance(),
                  "median": self.get_median(),
                  "max": self.get_max(),
                  "min":self.get_min()
                  }

            return pd.DataFrame.from_dict(data, orient="index", columns=self.features)
