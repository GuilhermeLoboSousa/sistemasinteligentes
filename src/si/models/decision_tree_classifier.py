import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
from typing import Literal, Tuple, Union

import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy
from src.si.statistics.impurity import gini_impurity, entropy_impurity


class Node:
    """
    Class representing a node in a decision tree.
    """

    def __init__(self, feature_idx: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None,
                 info_gain: float = None, value: Union[float, str] = None) -> None:#treshold é o que permmite decidir se vamos para esq ou direita
        #value apenas tem para as folhas, final da arvore!!!!!
        #info gain ganho que diz qual split é o melhor
        """
        Creates a Node object.

        Parameters
        ----------
        feature_idx: int
            The index of the feature to split on.
        threshold: float
            The threshold value to split on.
        left: Node
            The left subtree.
        right: Node
            The right subtree.
        info_gain: float
            The information gain.
        value: Union[float, str]
            The value of the leaf node.
        """
        # for decision nodes - para nos no meio
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf nodes
        self.value = value #apenas para folhas


class DecisionTreeClassifier:
    """
    Class representing a decision tree classifier.
    """

    def __init__(self, min_sample_split: int = 2, max_depth: int = 10,
                 mode: Literal['gini', 'entropy'] = 'gini') -> None:
        """
        Creates a DecisionTreeClassifier object.

        Parameters
        ----------
        min_sample_split: int
            minimum number of samples required to split an internal node.
        max_depth: int
            maximum depth of the tree.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain.
        """
        # parameters
        self.min_sample_split = min_sample_split #estamos a dizer neste caso que se o no tiver 2 samples fazemos split do no, mas poderia ser um numero mais alto e tornar este processo mais rapido mas provavelmente menos accurate
        self.max_depth = max_depth #niveis que vao existir na arvore
        self.mode = mode #aquela métrica pela qual escolhemos avaliar, que normalmente é um hiperparametro

        # estimated parameters
        self.tree = None
        self.dataset = None

    def _build_tree(self, dataset: Dataset, current_depth: int = 0) -> Node: #current_depth é o ponto de partida da arvore
        """
        Builds a decision tree recursively.

        Parameters
        ----------
        dataset: Dataset
            The dataset to build the tree on.
        current_depth: int
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the tree.
        """
        n_samples = dataset.shape()[0]
        if n_samples >= self.min_sample_split and current_depth <= self.max_depth: #aqui estamos a dar as 2 condições de paragem , tem de acontecer as duas
            #se o n_samples for maior que ou igual que o minimo entao há split caso contrario fica assim
            # tambem aqui vemos até que profundidade quermos avançar na arvore
            #quanto mais baixo o nº samples permitido(2) e maior a profundidade da arvore provavelmente mais tempo demora 
            best_split = self._get_best_split(dataset) # como se verificou as duas condições fazemos o split
            if best_split['info_gain'] > 0:#situação perfeito o info_gain dar 0 ou seja nao ganhamos mais nenhum info por isso acertamos todas as classes, logo dá logo o valor da folha
                #caso contrario vai correr até samples e perfundidade o permitirem
                left_subtree = self._build_tree(best_split['left_data'], current_depth + 1) #continua a fazer a mesma coisa recursivamente pela arvore a baixo e para cada feature
                right_subtree = self._build_tree(best_split['right_data'], current_depth + 1)
                return Node(best_split['feature_idx'], best_split['threshold'], left_subtree, right_subtree,
                            best_split['info_gain'])
        leaf_value = max(dataset.y, key=list(dataset.y).count) #atenção que aqui o dataset.y é dos dados do no final
        #pode ser complciado de perceber mas isto é recursivo logo ele vai ajustando os dados a medida que vai descendo na arvore ate uma das condições do if se verificar
        return Node(value=leaf_value)#apenas para a folha

    def _get_best_split(self, dataset: Dataset) -> dict:
        """
        Finds the best split for a dataset based on the information gain.

        Parameters
        ----------
        dataset: Dataset
            The dataset to find the best split for.

        Returns
        -------
        dict
            A dictionary containing the best split containing the feature index, threshold, left and right datasets,
            and the information gain.
        """
        best_split = {} # queremos que tenha os features index que estao associados ao split, o datasset da esquerda e da direita e o treshold que é a regra do plit
        info_gain = float('-inf')
        for feature_idx in range(dataset.shape()[1]):#vai percorrer feature a feature
            features = dataset.X[:, feature_idx] #todas as samples daquela coluna, ou seja estamos a analisar coluna a coluna
            possible_thresholds = np.unique(features) #tresh hold possiveis sao os valores de feature para cada sample
            for threshold in possible_thresholds[:-1]: #vai ver qual o melhor, ou seja todos os valores existentes de uma coluna menos o ultimo
                #o ultimo já nao adianta fazer porque basicamente é o que sobra
                left_data, right_data = self._split(dataset, feature_idx, threshold)#aqui fazemos o splite com todos os treshold possiveis menos 1
                y, left_y, right_y = dataset.y, left_data.y, right_data.y #preparar para calcular impurity, onde se usa os y information
                current_info_gain = self._information_gain(y, left_y, right_y) #y sera sempre o os dados do no de cima e left e right os nos seguinte (pos split)
                if current_info_gain > info_gain:#tenho que ter sempre ganho que mostra o quao bem foi feito o split e aí guardo os dados
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_data': left_data,
                        'right_data': right_data,
                        'info_gain': current_info_gain
                    }
                    info_gain = current_info_gain #vou atualizar, permite depois iteração apos iteração saber qual o melhor
        # check if best split is not empty (cases where labels of the dataset are all the same)
        if not best_split:
            best_split = {'info_gain': info_gain}
        return best_split

    @staticmethod
    def _split(dataset: Dataset, feature_idx: int, threshold: float) -> Tuple[Dataset, Dataset]:
        """
        Splits a dataset into left and right datasets based on a feature and threshold.

        Parameters
        ----------
        dataset: Dataset
            The dataset to split.
        feature_idx: int
            The index of the feature to split on.
        threshold: float
            The threshold value to split on.

        Returns
        -------
        Tuple[Dataset, Dataset]
            A tuple containing the left and right datasets.
        """
        left_indices = np.argwhere(dataset.X[:, feature_idx] <= threshold).flatten()#flatten é para passsar de [[]] para []
        #indices/posições das samples com menor valor, para a feature em análise, do que o treshold (esquerda)
        # mesma coisa para a direita mas com mior valor que treshold 
        right_indices = np.argwhere(dataset.X[:, feature_idx] > threshold).flatten()
        #esquerda vai ficar as samples que têm menos que um treshold e direita as que tem mais
        left_data = Dataset(dataset.X[left_indices], dataset.y[left_indices], features=dataset.features,
                            label=dataset.label) #apenas fico com as samples da esqueda 
        right_data = Dataset(dataset.X[right_indices], dataset.y[right_indices], features=dataset.features,
                             label=dataset.label) #apenas fico com as sample da direita
        return left_data, right_data

    def _information_gain(self, parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
        """
        Calculates the information gain of a split.
        It can be used for both gini and entropy.

        Parameters
        ----------
        parent: np.ndarray
            The parent data.
        left_child: np.ndarray
            The left child data.
        right_child: np.ndarray
            The right child data.

        Returns
        -------
        float
            The information gain of the split.
        """
        weight_left = len(left_child) / len(parent) #peso do split feito a esquerda
        weight_right = len(right_child) / len(parent)#peso feito a direita
        #tem que ser pq podemos ter um split que deu 50 para a esquera e 5 para a direita logo nao faz sentido terem ambos o mesmo peso
        if self.mode == 'gini':
            return gini_impurity(parent) - (weight_left * gini_impurity(left_child) + weight_right * gini_impurity(right_child))
        #penso que queremos um giniimpurtity pequenoe e ntropia pequeno
        elif self.mode == 'entropy':
            return entropy_impurity(parent) - (weight_left * entropy_impurity(left_child) + weight_right * entropy_impurity(right_child))
        else:
            raise ValueError(f'Invalid mode: {self.mode}. Valid modes are: "gini", "entropy"') #apenas pode ser esses dois

    def print_tree(self, tree: Node = None, indent: str = '\t') -> None:
        """
        Prints the decision tree.

        Parameters
        ----------
        tree: Node
            The root node of the tree.
        indent:
            The indentation to use.
        """
        if not tree:
            tree = self.tree
        if tree.value is not None:
            print(tree.value)
        else:
            print(f'{self.dataset.features[tree.feature_idx]} <= {tree.threshold}')
            print(f'{indent}left: ', end='')
            self.print_tree(tree.left, indent + '  ')
            print(f'{indent}right: ', end='')
            self.print_tree(tree.right, indent + '  ')

    def fit(self, dataset: Dataset) -> 'DecisionTreeClassifier':
        """
        Fits the decision tree classifier to a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        DecisionTreeClassifier
            The fitted model.
        """
        self.dataset = dataset
        self.tree = self._build_tree(dataset)
        return self

    def _make_prediction(self, x: np.ndarray, tree: Node) -> Union[float, str]:
        """
        Makes a prediction for a single sample.

        Parameters
        ----------
        x: np.ndarray
            The sample to make a prediction for.
        tree: Node
            The root node of the tree.

        Returns
        -------
        Union[float, str]
            The predicted value.
        """
        if tree.value is not None:
            return tree.value #se estou numa folha quero saber que classe corresponde, ou seja , num caso binario posso ter 10-0 e 2-1 logo o valor é classe 0
        feature_value = x[tree.feature_idx] #tem que ser visto com a função  e nao esqueçamos que aqui a arvore ja esta feita
        if feature_value <= tree.threshold:#menor ou igual sabemos que olhamos para a esq.
            #esta parte permite verificar as prediciton ao longo da arvore ou seja vamos imaginar que estamos no nó do meio permite descer ate uma folha
            return self._make_prediction(x, tree.left) #o x vai sempre atualizando (vemos isso na função de baixo)
        else:
            return self._make_prediction(x, tree.right) #percorre sempre a arvore ate atingir uma folha e ai da o valor
        #suma:
        #arvore esta feita e cada folha tem uma classe associada,"classe vencedora"
        #agora cada sample do test vai percorrer a arvore e calhar numa folha, e fica com a classe associdada
        #no fim temos x classes associadas, onde x = numero de samples do teste
        #portanto podemos ter algo assim [1,1,1,0,1] e depois o que fazemos é o score 
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions for.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        predictions = [self._make_prediction(x, self.tree) for x in dataset.X]#aqui vai permmitir ver a sample A,B,C
        return np.array(predictions) #fico com um array de prediciton[0,1,1,1,0,1] neste caso 5 samples testadas

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
    model = DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')
    model.fit(train)
    model.print_tree()
    print(model.score(test))


    #nota de raciocinio:
    #caso o dataset seja muito grande e a arvore sempre a aumentar a profundidade, ou seja, ganhamos sempre com o split corremos o risco de ter algo infinito. dai a escolha do numero de profundiadde e de samples permitido

    #suma do que percebi:
    #1-vai percorrer todas as features e basicamente garantir que o split feito com uma determinada feature e um determinado treshold é o que da mais ganho;
    #2-faz isso sempre que o split der ganho , ou ate n_samples e profundidade o permitirem
    #3- caso o split nao der ganho chegamos a uma folha
    #4-temos a arvore desenhada
    #5agora é pegar em samples e percorrer a arvore
    #6-vamos ter um array de previsoes que devemos comparar com o original
    #7- temos a ccuracy do modelo
