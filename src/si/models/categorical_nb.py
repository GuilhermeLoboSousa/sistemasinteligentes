import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.metrics.accuracy import accuracy
from src.si.metrics.rmse import rmse
from src.si.model_selection.split import stratified_train_test_split
from typing import Callable,Union
from src.si.model_selection.split import train_test_split

class CategoricalNB:
    """
    """

    def __init__(self,smothing:float=1.0): #O objetivo do Laplace Smoothing é adicionar uma pequena quantidade (normalmente um) a todas as contagens . Esa "suavização" ajuda a resolver problemas em que uma probabilidade de zero ou uma contagem de zero .
        self.smothing=smothing

    def class_prior(self,dataset:Dataset):
        classes,count=np.unique(dataset.y,return_counts=True)
        total=dataset.shape()[0]
        prob_por_classe=[]
        for cada_classe in classes:
            adicionar=(count[cada_classe] + self.smothing)/total
            prob_por_classe.append(adicionar)
        return prob_por_classe
    
def feature_probs(self, dataset: Dataset):
    classes = np.unique(dataset.y)
    num_classes = len(classes)
    num_features = dataset.X.shape[1]  # Número de recursos

    # Inicialize um dicionário para armazenar as probabilidades de cada recurso
    feature_probabilities = {feature_idx: {0: np.zeros(num_classes), 1: np.zeros(num_classes)} for feature_idx in range(num_features)}

    for i, classe in enumerate(classes):
        # Filtra o dataset apenas para a classe atual
        class_subset = dataset[dataset.y == classe]

        for feature_idx in range(num_features):
            # Contagem de exemplos com o recurso sendo 0
            count_0 = np.sum(class_subset.X[:, feature_idx] == 0)
            # Contagem de exemplos com o recurso sendo 1
            count_1 = np.sum(class_subset.X[:, feature_idx] == 1)

            # Armazena as contagens nas probabilidades
            feature_probabilities[feature_idx][0][i] = count_0
            feature_probabilities[feature_idx][1][i] = count_1

    return feature_probabilities


    
