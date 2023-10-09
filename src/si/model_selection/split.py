import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset


def train_test_split(dataset:Dataset,test_size:float=0.2,random_state:int=42) -> tuple:
    """
    Split the dataset into training and testing sets according a certain percentage and in a random way

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """

    np.random.seed(random_state) #possibilita gerar numeros aleatorios , mas que esses numeros consigam ser repetidos-1 coisa a se fazer
    
    n_samples=dataset.X.shape[0] #numero de linhas do meu dataset
    permutations=np.random.permutation(n_samples) #algo do genero[4,3,7,8] seleciona as linhas aleatoriamente

    n_test=int(n_samples*test_size)#saber com quantas linhas vamos ficar o que depende obviamente do test_size ou seja se é 20 % ou 30 %,etc. O int é para arredondar
    
    test_index=permutations[:n_test]
    train_index=permutations[n_test:] #por norma o de treino é maior logo tem de ser assimd

    train=Dataset(dataset.X[train_index],dataset.y[train_index],features=dataset.features, label=dataset.label) #apenas muda X e y , ajustamos a treino e a test
    test=Dataset(dataset.X[test_index],dataset.y[test_index],features=dataset.features, label=dataset.label)

    return train,test


