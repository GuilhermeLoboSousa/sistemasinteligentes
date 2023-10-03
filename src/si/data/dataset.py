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

    def summary(self) -> pd.DataFrame:
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
    
    def dropna(self) -> np.ndarray:
        """
        this fuction allow to remove the lines that have nan values and behind that give return the index of this lines
        """
        without_na=np.isnan(self.X).any(axis=1)
        self.X=self.X[~without_na]
        self.y=self.y[~without_na]
        indice=np.where(without_na)
        return self,indice
    
    def fill_na(self, strategy:str=None):
        """
        this function allow replace de null values by the :
        first option: a value random choose betwween the min and max value of the column with nan values
        second: the median value of the column
        third: the mean value of the collumn
        Arguments
        -------
        strategy: we can choose if want to replace by the min or max, median or mean
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
        """
        if not isinstance(index, int):
            raise ValueError("Please provide a valid integer index.")
        
        if index <0 and index> self.X.shape[0]:
            raise ValueError("Put a valid index")
        
        self.X = np.delete(self.X, index, axis=0) #remover algo do eixo linhas do dataset X e esse algo é a posição index
        if self.y is not None:#caso exista y vou quere apagar tb o seu index
            self.y = np.delete(self.y, index)
        return self


    



#testar
# Dados de exemplo
X = np.array([[1, 2, 3],
              [4, 5, np.nan],
              [7, np.nan, 9]])
y = np.array([0, 1, 0])
features = ['feature_1', 'feature_2', 'feature_3']
label = 'target'

dataset = Dataset(X, y, features, label)
print("Shape:", dataset.shape())  # Deve retornar (3, 3)
print("Has Label:", dataset.has_label())  # Deve retornar True
print("Classes:", dataset.get_classes())  # Deve retornar [0 1]
print("Mean:", dataset.get_mean())  # Deve retornar [4. 5. 6.]
print("Variance:", dataset.get_variance())  
print("Median:", dataset.get_median())  # Deve retornar [4. 5. 6.]
print("Max:", dataset.get_max())  # Deve retornar [7. 8. 9.]
print("Min:", dataset.get_min())  # Deve retornar [1. 2. 3.]
summary_df = dataset.summary()
print("Summary:")
print(summary_df)
#removed_indices = dataset.dropna()
#print("dataset,Indices removidos:", removed_indices)  # Deve retornar os índices das linhas com valores NaN
#print("Shape após a remoção de NaNs:", dataset.shape())  # Deve retornar a nova forma após a remoção
strategy = "value"
dataset_filled_value = dataset.fill_na(strategy)
print(dataset_filled_value.X)
print(dataset.remove_by_index(0))
print(dataset.X,dataset.y)