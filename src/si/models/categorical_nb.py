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
        self.class_prior=None
        self.feature_probs=None

    
    def fit(self, dataset:Dataset) -> "NB":
        n_samples=dataset.X.shape[0] # numero de linhas
        n_features=dataset.X.shape[1] #numero de colunas
        unique_classes = np.unique(dataset.y) #possiveis classes que existem
        num_classes=len(unique_classes) #quantas classes sao

        class_counts=np.zeros(num_classes)
        feature_counts=np.zeros((num_classes,n_features))
        self.class_prior=np.zeros(num_classes)

        # Calculate class_counts 
        for class_label in range(num_classes): # quere saber quntas samples tenho por classe, mas adicionar o var para n ter zeros 
            class_counts[class_label] = np.sum(dataset.y == class_label) + self.smothing
        
        # Calculate feature_counts , saber quantas classe 0 ou 1 tenho por coluna
        for class_label in range(num_classes):
            class_samples = dataset.X[dataset.y == class_label] #fico no dataset apenas com classe 0 ou 1 ( no caso exemplo)
            feature_counts[class_label, :] = np.sum(class_samples, axis=0) + self.smothing #começa na primeira linha e regista o somatorio por coluna de todas

        self.class_prior=class_counts/n_samples #probabilidade de pertencer a classe 0 ou 1, difente do class_count que é frequencia absoluta

        self.feature_probs = np.zeros((num_classes, n_features), dtype=float) #probabilidade de cada feature ser de classe 0 ou 1 , face ao eexmplo em estudo
        #seria uma matriz de n_class pelo numero de features

        # Calculate feature_probs , realizar o que está no slide 27 da aula 4
        for class_label in range(num_classes):
            self.feature_probs[class_label, :] = feature_counts[class_label, :] / class_counts[class_label]  #cada valor vai ser divido pelo total 

        return self

    def predict (self,dataset:Dataset):
        """
        """
        if self.feature_probs is None or self.class_prior is None:
            raise ValueError("O modelo não foi treinado. Use fit() para treinar o modelo primeiro.")
        
        n_samples = dataset.X.shape[0]
        num_classes = len(self.class_prior)

        class_probs = np.zeros((n_samples, num_classes), dtype=float) # matriz que vai ter linhas=n_samples e colunas = n_features, onde vamos er a probabilidade de cada sample pertencer a uma determinada class
        #sendo que vamos querer ficar com a maior probabilidade

        predicted_class = np.zeros(n_samples, dtype=int)  # para guardar os resultados final

        for sample_index in range(n_samples):#quero que calcule por todas as linhas, como diz no slide "for each sample"
            sample = dataset.X[sample_index]
            for class_label in range(num_classes): #como diz no slide "for each class"
                class_probs[class_label] = ( np.prod(sample * self.feature_probs[class_label] + (1 - sample) * (1 - self.feature_probs[class_label])) * self.class_prior[class_label]) #dado pelo slide do prof(nao chegava la)
            predicted_class[sample_index] = np.argmax(class_probs) #vamos ficar com um valor maximo por coluna que representa a probabilidade de pertencere a uma determinada classe
            #iremos obter o indice onde se obteve maior valor de class_probs, no nosso exemplo seria ou 0 ou 1
            #depois teremos de comparar o obtido com o real no dataset

        return predicted_class # algo com n_samples a 0 ou 1
    
    def score(self,dataset:Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        rmse: float
            The accuracy of the model
        """       
        predictions=self.predict(dataset)
        return accuracy(dataset.y,predictions) #comparar aquilo que foi obtido com o verdadeiro dataset


if __name__ == '__main__':
    X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 0, 0],
              [1, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [1, 1, 1],
              [0, 0, 1],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0],
              [1, 0, 0]])
    y = np.array([0,0,1,0,0,0,1,1,1,0,1,0,0,0,1])


    dataset_ = Dataset(X=X, y=y)

    #features and class name 
    dataset_.features = ["ceu","vento","humidade"]
    dataset_.label = "jogar"

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # regressor KNN
    nb = CategoricalNB(smothing=1.0)  

    # fit the model to the train dataset
    nb.fit(dataset_train)

    # evaluate the model on the test dataset
    score = nb.score(dataset_test)
    print(f'The accuracy of the model, made by me, is: {score}')


#teste com o do sckitlearn:
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Seu conjunto de dados
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 0, 0],
              [1, 1, 0],
              [0, 0, 1],
              [0, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [1, 1, 1],
              [0, 0, 1],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 0],
              [1, 0, 0]])
y = np.array([0,0,1,0,0,0,1,1,1,0,1,0,0,0,1])

# Variável para suavização (Laplace Smoothing)
var_smoothing = 1.0

# Aplicar suavização manualmente
X_smoothed = X + var_smoothing

# Separe os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_smoothed, y, test_size=0.2, random_state=42)

# Crie uma instância do CategoricalNB e ajuste-a aos dados de treinamento
nb = CategoricalNB()
nb.fit(X_train, y_train)

# Faça previsões usando os dados de teste
y_pred = nb.predict(X_test)

# Calcule a precisão das previsões
accuracy = accuracy_score(y_test, y_pred)

# Exiba a precisão
print("Precisão modelo feito por outros:", accuracy)
    
