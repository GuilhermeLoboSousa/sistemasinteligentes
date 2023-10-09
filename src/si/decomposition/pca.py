import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from typing import Callable
from src.si.data.dataset import Dataset
import matplotlib.pyplot as plt


class PCA:
    """
    """
    def __init__(self,n_components:int) -> None:
        """
        PCA algorithm that is used to reduce the dimensions of the dataset. For example PC1=loading1*coluna1 + loading2`coluna 2 ***
        Behind that we can for example not "use" all de pca , this is , in some cases with 3 or 4 PCA we already have a good representation of the data

        Parameters
        ----------------
        n_components:  Number of components to be considered and returned from the analysis.

        Atributtes
        ----------------
        mean
        components
        explained_variance

        """
        if n_components <= 1 :
            raise ValueError("O número de componentes principais (n_components) deve ser maior que 1.")
        self.n_components=n_components
        self.mean=None # média de cada samples
        self.components=None #componentes principais (matriz unitaria de eigenvectors)
        self.explained_variance=None # variancia explicada (matrix diagonal de eigenvalues)


    def fit(self, dataset:Dataset) -> tuple:
        """
        It fits the data and stores the mean values of each sample, the principial components and the explained variance.
        Parameters
        ----------
        dataset (Dataset): Dataset object.
        """
        #1-centering the data
        if self.n_components > dataset.X.shape[1]-1:#acrescentei estes avisos , que vao de acordo à teoria
            raise ValueError("O número de componentes principais não pode ser maior ou igual ao número de colunas no dataset.")
        self.mean=np.mean(dataset.X,axis=0) #quero a media das minhas samples, linha a linha portanto
        self.center=np.subtract(dataset.X,self.mean) #centrar os pontos

        #svd
        U,S,Vt=np.linalg.svd(dataset.X,full_matrices=False) # first step to find U: unitary matrix of eigenvectors; S: diagonal matrix of eigenvalues; Vt: unitary matrix of right singular vectors

        #principal components
        self.components=Vt[:self.n_components] # principais componentes sao os primeiros n_componentes de Vt

        # Explained Variance
        numero_samples=dataset.X.shape[0]
        EV=(S**2)/(numero_samples-1) # formula dada pelo professor
        self.explained_variance=EV[:self.n_components] #corresponder aos primeiros n_componentes de EV
        return self
    
    def transform (self,dataset:Dataset)->tuple:
        """
        Returns the calculated reduced Singular Value Decomposition (SVD)
        Parameters
        ----------
        dataset (Dataset): Dataset object
        """
        V=self.components.T #trnaposta da matrix Vt que estamos a trabalhar neste momento (ver acima)
        X_reduced=np.dot(self.center, V) #pc sao calculados a partir dos dados centrados
        return X_reduced
    
    def fit_transform(self,dataset:Dataset) -> tuple:
        """
        It fit and transform the dataset
        Parameters
        ----------
        dataset (Dataset): Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)
    
    def plot_variance_explained(self):
        """
        Creates  a bar plot of the variances explained by the principal components.
        """
        if self.explained_variance is not None:
            explained_variance_normalized = self.explained_variance / sum(self.explained_variance) #normalizar as variancias
            print(explained_variance_normalized)

            num_pcs = len(self.explained_variance) #preparar para o eixo do X onde vao os pc1,2,etc
            x_indices = range(1, num_pcs + 1)

            plt.bar(x_indices, explained_variance_normalized, align='center')
            plt.xlabel('Componente Principal (PC)')
            plt.ylabel('Variância Explicada Normalizada')
            plt.title('Variância Explicada por Componente Principal')
            plt.xticks(x_indices,[f'PC{i}' for i in x_indices])
            plt.show()
        else:
            print("Os componentes principais e as variâncias explicadas ainda não foram calculados.")        

#para acompar a parte teorica vi o video https://www.youtube.com/watch?v=FgakZw6K1QQ&t=10s

if __name__ == "__main__":
    # dados
    dataset = Dataset(X=np.array([[10, 21, 33,67],
                                  [4, 15, 16,43],
                                  [7, 28, 19,3],
                                  [10, 13, 27,22]]),
                      y=np.array([0, 1, 0, 1]),
                      features=["A", "B", "C","D"],
                      label="cancer")

    pca = PCA(n_components=3)
    transformed_data = pca.fit_transform(dataset)
    print("Conjunto de Dados Transformado:")
    print(transformed_data) #onde foi "projeto ao longo de pc1 -primeira colunca e ao longo de pc2-segunda coluna"
    #como nao percebia muito do ouput obtido decidi adicionar o metodo para ver a representabilidade
    print(pca.plot_variance_explained())




