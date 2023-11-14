import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import copy
from abc import abstractmethod

import numpy as np

from src.si.neural_networks.optimizers import Optimizer


class Layer: #cada camda de neurónio
    """
    Base class for neural network layers.
    """

    @abstractmethod# permite que metodos dessa classe sejam usados pela classe que utiliza a classe Layer neste caso
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input, i.e., computes the output of a layer for a given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error, i.e., computes dE/dX for a given dE/dY and update
        parameters if any.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        raise NotImplementedError

    def layer_name(self) -> str:
        """
        Returns the name of the layer.

        Returns
        -------
        str
            The name of the layer.
        """
        return self.__class__.__name__ #apenas vai buscar o nome da classe

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, shape: tuple):
        """
        Sets the shape of the input to the layer.

        Parameters
        ----------
        shape: tuple
            The shape of the input to the layer.
        """
        self._input_shape = shape

    def input_shape(self) -> tuple:
        """
        Returns the shape of the input to the layer.

        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self._input_shape

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        raise NotImplementedError


class DenseLayer(Layer): #vai buscar o abstract da layer , por ter denselayer(layer)
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__() #nao percebo bem pq esta aqui- talvez pelos abstract
        self.n_units = n_units #nº neuronios que queremos
        self._input_shape = input_shape #na primeira é o nº de features, tuple apenas para ficar tabular
        #estimated parameters
        self.input = None
        self.output = None
        self.weights = None #conectam com a layer seguinte
        self.biases = None #cada neuronio tem um bias associado

    def initialize(self, optimizer: Optimizer) -> 'DenseLayer': #ter pesos e bias random que depois vao sendo otimizados
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5 #estou a criar uma matriz de linhas=features, colunas = neuronios, ou seja vou ficar com features*neuronios=numero de pesos
        #linha feature-f1[f1neuronio1,f1neuronio2,etc] ou seja tenho um peso para a conexao da feature 1 com o neuronio 1 (da segunda layer), peso para feature 1 e neuronio 2(sgunda layer) e por ai fora
        #0.5 é picuinhas
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units)) #zeros de tamanho numero de neuronios, pq vou ter um bias por cada neuronio
        self.w_opt = copy.deepcopy(optimizer)#confirmar com o prof , mas penso que permite depois utilizar diferentes optimizers
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:#numero de pesos+bias: (peos= linhas*colunas) + bias 
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape) #np prod ja faz a multiplicação; nao esquecer que no caso do biases é 1* n neuronios(n_units)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:#boleano para saber se estamos ou nao em treino(util depois)
        # conta1=pesos*X + bias progressao linear para a layer seguinte
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input # nosso X no caso da primeira layer
        self.output = np.dot(self.input, self.weights) + self.biases #seguir a formula
        #apenas confirmar o self.output sera o input da layer seguinte certo?
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> float:
        #depois nadamos para tras para corrigir os pesos e os bias
        #trabalhamos com as derivadas
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        #erro que temos de descobrir para passar a layer anterior, sabendo que temos o output de erro como entrada
        input_error = np.dot(output_error, self.weights.T) #imput error de tamanho = ao  numero de neuronios da camada anterior
        # computes the weight error: dE/dW = X.T * dE/dY
        #erro associado ao pesos que temos de descobrir para passar a layer anterior 
        weights_error = np.dot(self.input.T, output_error) #seguir formula- input é algo por exemplo para 3 neuronios de [feature1,feature2,f3]
        # weight_error- tem de ser igual ao numero de pesos existetnees entre penultima camada e a camada seguitne
        # computes the bias error: dE/dB = dE/dY
        #errros associado ao bias que temos de desbocrir para passar atras
        bias_error = np.sum(output_error, axis=0, keepdims=True)#cutilizamos o sum porque o nosso outputerro pode nao ser so apenas [[0.3,0.7]] e ter mais como no caso da softmax
        #bias_error igual ao numero de neuronios da ultima camada
        #nao esquecer que no np.dot(A,B)- ncolunas A =nlinhas B
        #
        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error) #criado depois, para fazer optimização e respetivo update
        self.biases = self.b_opt.update(self.biases, bias_error)#p0odemos utilizar o gradiente descente como outros optimizerrs
        return input_error

    def output_shape(self) -> tuple: #tuple para ficar tabelar
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,) #numero de neuronios


