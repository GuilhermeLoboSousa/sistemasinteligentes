import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from typing import Callable
from src.si.data.dataset import Dataset
from src.si.statistics.euclidean_distance import euclidean_distance
import matplotlib.pyplot as plt


class Kmeans:
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point like [0,1,2,0] we now that the line 1 and 4 are associated with de cluster 0

    Parameters
    ----------
    k: int
        Number of clusters.
    max_iter: int
        Maximum number of iterations.
    distance: Callable
        Distance function.

    Attributes
    ----------
    centroids: np.array
        Centroids of the clusters.
    labels: np.array
        Labels of the clusters.
    """
    def __init__(self,k:int,max_iter:int=1000,distance:Callable=euclidean_distance):
        """
        same logic that before, so this is k-means clustering algotirhm
        """
        self.k=k
        self.max_iter=max_iter
        self.distance=distance

        self.centroids=None #coordenadas dos pontos medios (ver video)
        self.labels=None #classe que corresponde cada centroide

    def _init_centroides(self,dataset:Dataset):
        """
        So this functon generate thi initial k centroids, that is random (but "inside" data)
        
        Parameters
        --------
        Dataset:we pretende to star in a random way but inside our data
        """
        random_centroids=np.random.permutation(dataset.shape()[0])[:self.k] #permutação/troca aleatoria de uma matriz ou de um numero, onde neste caso apenas vemos os indices das linhas que sao lá está trocados de forma aleatoria, e depois apenas escolhemos os primeiros k
        #duvida é porque no 0 e nao no 1 , ou seria indiferente
        self.centroids=dataset.X[random_centroids] #LÁ esta seria indiferente desde que aqui tivesse isso em consideração ou seja se tivesse usado antes [1] agor teria de colocar[:,random_centroids]
        #ou seja aqui já tenho k pontos dos nossos dados , aleatorios que serao os centroides de ponto de partida

    def _get_closest_centroid(self,sample:np.ndarray) -> np.ndarray:
        """
        allow to obtain the closest centroid of each data point/sample
        """
        centroids_distance=self.distance(sample,self.centroids) #medir a distancia entre um determinado dado do meu dataset(sample) e o respetivo centroid, com o intuito mais à frente de minimizar esta distancia
        closest_centroid_index=np.argmin(centroids_distance,axis=0) #apenas vou querer ficar com o indice que esta associado à menor distancia; tem de ser axis=0 poruqe:
        #outout seria algo ([1, 2, 3]), logo nos queremos procurar em coluna e nao em linha , porque em linha apenas seria 1,2,3 e em colunas já conseguimos ver que o centroide mais proximo seria o primeiro 
        return closest_centroid_index #ou seja fazemos associação sample a um centroid (o mais perto e tiramos o indice desse centroid a que pertence a sample em questao)
    
    def fit (self,dataset:Dataset)->"Kmeans":
        """
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.
        """
        self._init_centroides(dataset) # primeiro passo atribuir centroides inicias

        convergence=False #nossa flag
        i=0 #contador
        labels=np.zeros(dataset.shape()[0]) # apenas para sabermos quantas labels temos que incialmente serao zeros, ou seja o mesmo nº de eros que o mesmo nº e linhas
        #situação do dataset iris o ideal era apareceer labels para cada centroid com apenas 1 especie em cada
        while not convergence and i<self.max_iter: #so para se ocorrer convergencia(centroids mais pertos), ou fazedno isto n vezes
            new_labels=np.apply_along_axis(self._get_closest_centroid,axis=1,arr=dataset.X) #vamos considerar cada linha, daí o eiixo =1, do dataset.X como sample e aplicar a função get_closest onde iremos obter uma lista com indices que representa sample pertence a um centroid com indice tal
            #[0,1,1,0] a primeira linha pertence ao primeiro centroid e por ai fora

            #vamos experimentar com novos centroides
            centroids=[]
            for j in range(self.k):#k centroides
                centroid=np.mean(dataset.X[new_labels == j],axis=0)#vai ser por exemplo datasset.X[True, False, True,True], entao ele já vai considerar a media para um novo centroid de todos os pontos que antigamente pertenciam a esse cluster
                #ou seja o false aqui já nao era considerado para "atrapalhar este nova media e repetivo novo cluster"
                #em suma "divide" o dataset em linhas que ja foram associadas a um cluster e calcula sua media, já é mais "filtrado"
                centroids.append(centroid)

           #transformar num array para poder trabalhar 

            self.centroids=np.array(centroids) #agora os self.centroids serao a media das colunas de uma determinada linha que já foi associada a um determinado cluster
            #deixa de o meu centroid ser um ponto random dos meus dados, para ser a media dos pontos que foram associdados a um cluster

            convergence=np.any(new_labels != labels) # se todos foram iguais retorna false e acaba o loop

            labels=new_labels #caso haja algo novo entao o vetor que ao inciao é de zero muda 
            i+=1 #passa ao seguinte

        self.labels=labels #guardamos as novas labels
        return self
    
    def _get_distance(self,sample:np.ndarray)->np.ndarray: #ja estava feito atras mas nao num metodo
        """
        distance between sample and the closest centroid
        """
        return self.distance(sample,self.centroids)
    
    def transform(self,dataset:Dataset)-> np.ndarray:
        """
        calculate the distance betwwen each sample and the closest centroid~
        """
        centroids_distance=np.apply_along_axis(self._get_distance,axis=1,arr=dataset.X) #ver linha a linha do dataset
        return centroids_distance #distancia de cada sample linha do dataset ao cluster
    
    def fit_transform(self,dataset:Dataset) -> np.ndarray:
        """
        It fits and transforms the dataset
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    def predict(self,dataset:Dataset)-> np.ndarray:
        """
        It predicts the labels of the dataset.
        """
        return np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)
    
    def fit_predict(self,dataset:Dataset)-> np.ndarray:
        """
        It fits and predicts the labels of the dataset.
        """
        self.fit(dataset)
        return self.predict(dataset)

if __name__ == '__main__':
    np.random.seed(0)
    num_samples = 100
    num_features = 5
    X = np.random.rand(num_samples, num_features)  # Dados de recursos aleatórios
    y = np.random.randint(0, 2, num_samples)  # Rótulos aleatórios (assumindo uma classificação binária)

    dataset = Dataset(X=X, y=y)
    elbow=[]
    for k_ in range (2,101):
        kmeans = Kmeans(k_)
        res = kmeans.fit_transform(dataset)
        predictions = kmeans.predict(dataset)
        #print(res.shape,"oi")
        #print(predictions.shape)
        distances = kmeans.transform(dataset)
        close_dist = np.min(distances, axis=1)
        cotovelo=np.mean(close_dist,axis=0)
        elbow.append(cotovelo)
    
    # Plote a curva da variância em relação a k
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 101), elbow, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Variancia das Percentagens')
    plt.title('Método do Cotovelo (Elbow Method)')
    plt.show()
         #cotovelo onde a distancias devem ir diminuindo, ou seja ele faz media de todas as distancias mais proximas e é normal se k = 100 que a distancia seja zero

        # unique_values, counts = np.unique(teste, return_counts=True)
        # total_samples = len(teste)
        # percentages = (counts / total_samples) * 100
        # for value, count, percentage in zip(unique_values, counts, percentages):
        #     print(f"Valor {value}: {count} ocorrências ({percentage:.2f}%)") 

    
    

