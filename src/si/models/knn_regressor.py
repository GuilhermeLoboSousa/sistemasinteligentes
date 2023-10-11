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
    def __init__(self,k:int=1,distance:Callable=euclidean_distance):
        """
        This algorithm predicts the class for a sample using the k most similar examples.But is suitable for regression problems.
        So estimates the average value of the k most similar examples instead of the most common class.
        Args:
            k :int 
                number of examples to consider
            distance: Callable 
                euclidean distance function. .
        """
        self.k=k # numero de k-mais proximos exemplos a considera
        self.distance=distance #função que vai calcular a distancia entre uma samples e as samples do dataset treino
        self._train_dataset=None # meu dataset de treino

    def fit(self, dataset:Dataset) -> "KNNRegressor":
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self._train_dataset=dataset # o input é o dataset de treino logo apenas fiz este passo-guardar o dataset treino
        return self
    
    def _get_closest_value_label(self,sample:np.ndarray)->int:
        """
        """
        distances=self.distance(sample,self._train_dataset.X) #calcular a distancia entre cada sample e o conjunto de sample no dataset de treino
        k_nearest_neighbors=np.argsort(distances)[:self.k] #argsort coloca as distancias por ordem crescente, logo quero as primeiras k -vao ser as distancias mais perto
        #k_nearest_neighbours cporresponde aos indices das samples com distancia mais proxima a sample de input
        k_nearest_neighbors_values_labels=self._train_dataset.y[k_nearest_neighbors] #  descubro que valores em Y corresponde a esses indices de mais curta distancia
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

