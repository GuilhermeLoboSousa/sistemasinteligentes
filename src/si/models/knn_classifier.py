import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.metrics.accuracy import accuracy
from typing import Callable,Union
from src.si.model_selection import split



class KNNClassifier:
    """
    KNN Classifier
    The k-Nearst Neighbors classifier is a machine learning model that classifies new samples based on
    a similarity measure (e.g., distance functions). This algorithm predicts the classes of new samples by
    looking at the classes of the k-nearest samples in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init(self,k:int=1,distance:Callable=euclidean_distance):
        """
        Initialize the KNN classifier

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """

        self.K=k # numero de k-mais proximos exemplos a considera
        self.distance=distance #função que vai calcular a distancia entre uma samples e as samples do dataset treino
        self.train_dataset=None #meu dataset treino

    def fit (self,dataset:Dataset) -> "KNNClassifier":
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNClassifier
            The fitted model
        """
        self.train_dataset=dataset # o input é o dataset de treino logo apenas fiz este passo
        return self
    
    def _get_closest_label(self,sample:np.ndarray)->Union[int,str]:
        """
        """
        distances=self.distance(sample,self.train_dataset.X) #calcular a distancia entre cada sample e o conjunto de sample no dataset de treino
        k_nearest_neighbors=np.argsort(distances)[:self.k] #argsort coloca as distancias por ordem crescente, logo quero as primeiras k -vao ser as distancias mais perto
        #k_nearest_neighbours cporresponde aos indices das samples com distancia mais proxima a sample de input
        k_nearest_neighbors_labels=self.train_dataset.y[k_nearest_neighbors] #  descubro que classes em Y corresponde a esses indices de mais curta distancia
        #ou seja vou ficar a saber quais classes da label y estao mais proximas
        labels, counts = np.unique(k_nearest_neighbors_labels, return_counts=True) #permite contar quantas vezes aparece cada classe mais proxima identificada anteriormente do genero (classe 0,3) (classe 1,2) (classe 2,1)
        return labels[np.argmax(counts)]#apenas quero a que aparece mais vezes , logo vou procurar o maximo
    
    def predict(self,dataset:Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset
        Go to every sample and calculate the distance of each sample(line dataset) to the rest of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model, with the supose classes
        """
        return np.apply_along_axis(self._get_closest_label,axis=1,arr=dataset.X) #ou seja aplicar a função self._get_closest_label a todas as linhas do dataset.X-aqui ja nao é especifico do treino ou teste
        #aqui o sample é cada linha do datasset.X e depois aplica a função em questao linha a linha

    def score (self,dataset:Dataset) ->float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions=self.predict(dataset)
        return accuracy(dataset.y,predictions) #comparar aquilo que foi obtido com o verdadeiro dataset

if __name__ == '__main__':
    # import dataset

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the KNN classifier
    knn = KNNClassifier(k=3)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The accuracy of the model is: {score}')


