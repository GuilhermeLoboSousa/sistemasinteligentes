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

def stratified_train_test_split(dataset:Dataset,test_size:float=0.2,random_state:int=42) -> tuple[Dataset,Dataset]: #utilizado quando a proporção das classes nao é correta, ou seja garante que a proporção de cada classe é igual no dataset treino e test
    """
    Divide the dataset into training and testing sets in a stratified manner.

    Parameters
    ----------
    dataset: Dataset
        The dataset to be split.
    test_size: float
        The proportion of the dataset to be used for testing.
    random_state: int
        The seed for random number generation.

    Returns
    -------
    train: Dataset
        The training dataset.
    test: Dataset
        The testing dataset.
    """
    np.random.seed(random_state)

    labels, counts = np.unique(dataset.y, return_counts=True) #permite contar quantas vezes aparece cada classe mais proxima identificada anteriormente do genero (classe 0,3) (classe 1,2) (classe 2,1)
    train_index=[]
    test_index=[]
    contador_letras=0
    for class_label in labels:#ciclo for é necessario para ter a preocupação de que se mantem a proporção de cada classe quer no dataset treino como teste
        freq=counts[contador_letras]
        teste_samples=int(freq*test_size) #saber com quantas vamos ficar para teste e consequentemente para treino
        class_indices =np.where(dataset.y == class_label)[0]#verificar os indices onde se verifica a classe em questao no loop
        shuffle=np.random.permutation(class_indices) #fazer shuffle desses indices para colocar aleatoriedade
        select_indices_test=shuffle[:teste_samples] #selecionar alguns para teste e outros para treino, mantendo a proporção
        select_indices_train=shuffle[teste_samples:]
        print(select_indices_test)
        test_index.append(select_indices_test) #colcoar tudo numa lista
        train_index.append(select_indices_train) #colcoar tudo numa lista
        contador_letras+=1
    
    train=Dataset(dataset.X[train_index],dataset.y[train_index],features=dataset.features, label=dataset.label) #apenas muda X e y , ajustamos a treino e a test
    test=Dataset(dataset.X[test_index],dataset.y[test_index],features=dataset.features, label=dataset.label)
    return train,test



