import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.metrics.accuracy import accuracy
from src.si.metrics.rmse import rmse
from src.si.model_selection.split import stratified_train_test_split
from typing import Callable, Union, Literal
from src.si.model_selection.split import train_test_split

class KNNRegressor:
    """
    The k-Nearest Neighbors regressor is a machine learning model that predicts the value of a new sample based on 
    a similarity measure (e.g., distance functions). This algorithm estimates the value of a new sample by
    considering the values(mean) of the k-nearest samples in the training data.

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
    def __init__(self, k: int = 1, weights: Literal['uniform', 'distance'] = 'uniform',  distance: Callable = euclidean_distance):
        '''
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        weights: Literal['uniform', 'distance']
            The weight function to use
        distance: Callable
            The distance function to use
        '''
        # parameters
        self.k = k
        self.distance = distance
        self.weights = weights

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        '''
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        '''
        self.dataset = dataset
        return self
    
    def _get_weights(self, distances: np.ndarray) -> np.ndarray:
        '''
        It returns the weights of the k nearest neighbors

        Parameters
        ----------
        distances: np.ndarray
            The distances between the sample and the dataset

        Returns
        -------
        weights: np.ndarray
            The weights of the k nearest neighbors
        '''
        # get the k nearest neighbors (first k indexes of the sorted distances)
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the weights of the k nearest neighbors
        weights = 1 / distances[k_nearest_neighbors]
        return weights
    
    def _get_weighted_label(self, sample: np.ndarray) -> Union[int, str]:
        '''
        It returns the weighted label of the most similar sample in the dataset

        Parameters
        ----------
        sample: np.ndarray
            The sample to predict

        Returns
        -------
        label: Union[int, str]
            The weighted label of the most similar sample in the dataset
        '''
        # get the distances between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the weights of the k nearest neighbors
        weights = self._get_weights(distances)

        # get the k nearest neighbors (first k indexes of the sorted distances)
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # get the weighted label
        label = np.sum(k_nearest_neighbors_labels * weights) / np.sum(weights)
        return label
    
    def _get_closest_value_label(self,sample:np.ndarray)->int:
        """
        It returns the label of the most similar sample in the dataset

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label:  int
            The label of the most similar sample in the dataset
        """
        distances=self.distance(sample,self.dataset.X) #calcular a distancia entre cada sample e o conjunto de sample no dataset de treino
        k_nearest_neighbors=np.argsort(distances)[:self.k] #argsort coloca as distancias por ordem crescente, logo quero as primeiras k -vao ser as distancias mais perto
        #k_nearest_neighbours cporresponde aos indices das samples com distancia mais proxima a sample de input
        k_nearest_neighbors_values_labels=self.dataset.y[k_nearest_neighbors] #  descubro que valores em Y corresponde a esses indices de mais curta distancia
        return np.mean(k_nearest_neighbors_values_labels) #mesma logica mas agora aplico as medias
        #ou seja vou ficar com algo um pouco distinto na maneira em que terei por exemplo media de 1,25 associado a um determinado conjunto A de samples
        # media de 2,34 associado a um conjunto B... 
    
    def predict(self,dataset:Dataset) -> np.ndarray: # array de medias
        """
        It predicts the mean label values of the given dataset
        Go to every sample and calculate the distance of each sample(line dataset) to the rest of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model, with the supose values
        """
        return np.apply_along_axis(self._get_closest_value_label,axis=1,arr=dataset.X) #ou seja aplicar a função self._get_closest_label a todas as linhas do dataset.X-aqui ja nao é especifico do treino ou teste
        #aqui o sample é cada linha do datasset.X e depois aplica a função em questao linha a linha
        # ve distancias de cada sample(cada linha do dataset total) ao dataset treino
        # seleciona as k mais proximas e faz uma medias dos valores y dessas mais proximas
        # faz isso para tudo, para num cenarioideal conseguir prever um valor final e nao uma classe propriamente dita

    def score(self,dataset:Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        rmse: float
            The rmse(error) of the model
        """       
        predictions=self.predict(dataset)
        return rmse(dataset.y,predictions) #comparar aquilo que foi obtido com o verdadeiro dataset


if __name__ == '__main__':
    num_samples = 600
    num_features = 100

    X = np.random.rand(num_samples, num_features)
    y = np.random.rand(num_samples)  # Valores apropriados para regressão

    dataset_ = Dataset(X=X, y=y)

    #features and class name 
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "target"

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # regressor KNN
    knn_regressor = KNNRegressor(k=5)  

    # fit the model to the train dataset
    knn_regressor.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn_regressor.score(dataset_test)
    print(f'The rmse of the model is: {score}')

