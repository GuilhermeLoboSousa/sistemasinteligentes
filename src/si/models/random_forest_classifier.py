import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
from typing import Literal, Tuple, Union

import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy
from src.si.statistics.impurity import gini_impurity, entropy_impurity
from src.si.models.decision_tree_classifier import DecisionTreeClassifier
from collections import Counter

class RandomForestClassifier:
    """
    ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting
    """

    def __init__ (self,n_estimators:int=1000, 
                 max_features:int=None, 
                 min_sample_split:int=2,
                 max_depth:int=10,
                 mode: Literal['gini', 'entropy'] = 'gini',
                 seed:int=42):
        """
        Uses a collection of decision trees that trains on random subsets of the data using a random subsets of the features.

        Parameters
        ----------
        n_estimators: int
            number of decision trees to use.
        max_features: int
            maximum number of features to use per tree. #quantas features vou suar para treinar com diferentes minidataset
        min_sample_split: int
            minimum samples allowed in a split.
        max_depth: int
            maximumdepth of the trees.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain.
        seed: int
            random seed to use to assure reproducibility
        """
        self.n_estimators = n_estimators
        self.max_features = max_features   
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed=seed

        #estimated parameters
        self.trees=[]#arvores criadas e respetivas features,inicalizar a vazio


    def fit(self,dataset:Dataset)-> "RandomForestClassifier":
        """
        Fits the random forest classifier to a dataset.
        train the decision trees of the random forest
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        #seed importarte para garantir consistencia na aleatoriedade, ou seja os numeros aleatorios gerados serao os mesmos-reprodutibilidade e poder de comparação
        if self.seed is not None:
            np.random.seed(self.seed)
        n_samples, n_features = dataset.shape() #ficar com nº de samples e features
        if self.max_features is None:#CASO SEJA NULO A REGRA É APLICAR RAIZ QUADRADA E APROXIMAR A INTEIRO
            self.max_features=int(np.sqrt(n_features))
        #vamos criar o boostrap onde queremos samples escolhidas possam ser repetidas mas as features nao
        #ou seja eu vou ficar por exemplo com um dataset(samples a,b,c,d; features f1,f2,f3) apos boostrap de por exemplo (a-f2,a-f3,b-f1)isto aleatoriamente e iterado
        #permite treinar arvores para um enorma variedade de datset de treino,pergunta ao prof entao o bootsrap é como se eu tivesse inderetamente a aumentar as possibilidade de um dataset quase como se tivesse a introduziir novas samples?
        for x in range(self.n_estimators):#devo iterar o bootsrap quantas arvores eu quiser fazer
            bootstrap_samples_index= np.random.choice(n_samples, n_samples, replace=True)#escolher random samples com um tamanho de n_samples e com replacement-onde vai dar os indices escolhidos pq n_saples é tamho de samples e nao os reais valores
            bootstrap_features_index = np.random.choice(n_features, self.max_features, replace=False)
            random_dataset = Dataset(dataset.X[bootstrap_samples_index][:, bootstrap_features_index], dataset.y[bootstrap_samples_index])
            
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_samples_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(random_dataset)

            #tupple com indices das features utilizadas e a arvore treino
            self.trees.append((bootstrap_features_index, tree))
        return self
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class labels for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to make predictions.

        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        # Criar uma matriz para armazenar as previsões de cada árvore
        n_samples=dataset.shape()[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object)#matriz onde as linhas serao as arvores realizadas e as colunas a previsao destas arvores para cada sample

        # Para cada árvore
        contador=0 #permite adcionar a cada linha da matriz de zeros criada 
        for features_indices, tree in enumerate(self.trees):
            features_index=self.trees[features_indices][0]
            real_tree=self.trees[features_indices][1]
            #apenas vou querer analisar as previsoes das featues analisadas na arvore em questao(daí anteriormente ter guardado num tupple) 
            # logo nao faço predict do dataset todo mas apenas do dataset(input) nas features definidas 
            sampled_data = Dataset(dataset.X[:, features_index], dataset.y)
            
            tree_predictions = real_tree.predict(sampled_data) #agora sim estou a fazer o rpedict com apenas as features que a arvore em questao foi treinada
            predictions[contador, :] = tree_predictions  # Armazenar as previsões dessa árvore
            contador +=1
        # Agora, para cada sample, escolher a classe mais comum entre as previsões de todas as árvores, ou seja tenho de fazer a transposta par apoder ter por linha a previsao para cada sample
        por_linha=np.transpose(predictions)
        def most_frequent(arr):#auxiliar para descobrir a classe que aparece mais vezes
            counter = Counter(arr)#Counter cria dict class:frequencia 
            most_comum = counter.most_common(1)[0][0] #dá a classe mais comum poerite ir bucar a um tuple(class,freque) a class
            return most_comum
        #ir buscar a classe que aparece mais vezes em cada linha, ou seja a classe que foi associada a cada sample por mais arvores
        most_frequent_values = np.apply_along_axis(most_frequent, axis=1, arr=por_linha)


        return most_frequent_values

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    from src.io.csv_file import read_csv
    from src.si.model_selection.split import train_test_split
    filename = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\iris\iris.csv"

    data = read_csv(filename, sep=",",features=True,label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=10000,max_features=4,min_sample_split=2, max_depth=5, mode='gini',seed=42)
    model.fit(train)
    print(model.score(test))

#esta feito falta de alguma forma comprovar se esta bem e fazer com 10000 ou 1000 da o mesmo
    

    