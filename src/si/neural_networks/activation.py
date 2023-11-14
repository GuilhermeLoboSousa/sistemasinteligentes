import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

from abc import abstractmethod
from typing import Union

import numpy as np

from src.si.neural_networks.layers import Layer


class ActivationLayer(Layer):#ativaão tem umas poucas de funções chave, onde numa parte intermedia dao nao linearidade e na parte final permitem ter output dos dados
    #se for multiclasse sofmax, binaria pode ser sigmoid
    """
    Base class for activation layers.
    """

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
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
        self.input = input
        self.output = self.activation_function(self.input) #função com abstract que vai ser criada depois consoante aquilo que quermos , relu,sigmoide softmax etc
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error #derivada depende da função de ativação escolhihda e tb vai ser depois

    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError

    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0 #nada haver com parameter


class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input)) #<0.5 é zero >0.5 é 1

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))


class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input) #com x negativo o valor é zero, com x positivo o valor é o maximo
    #[0.3,0.4,-0.9] ficaria com [0.3,0.4,0]

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input > 0, 1, 0) #onde o input for positivo fica 1, caso contrario fica 0
    #[0.3,0.4,-0.9] ficaria com [1,1,0]

class Softmaxactivation(ActivationLayer):
    """
    Softmax activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        shifted_input = input - np.max(input, axis=0, keepdims=True)#vou ter de ver o max do x por cada coluna do input e tirar ao input esse valor
        #fico com um novo input penso que seja para evitar altos valores exponenciais 
        # Compute the exponentials of the shifted input
        exp_input = np.exp(shifted_input)

        # Compute the softmax output
        softmax_output = exp_input / np.sum(exp_input, axis=0, keepdims=True) #formula

        return softmax_output

    def derivative(self, input: np.ndarray):
        """
        Derivative of the Softmax activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input)*(1-self.activation_function(input))
    
class TanhActivation(ActivationLayer):
    """
    Tanh activation function.
    """

    def activation_function(self, input: np.ndarray):
        """
        Tanh  activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        numerador=np.exp(input)-np.exp(-input) #seguir formula
        denominador=np.exp(input) + np.exp(-input)
        return numerador/denominador

    def derivative(self, input: np.ndarray):
        """
        Derivative of the Tanh  activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        tanh_output = self.activation_function(input)
        return 1 - (tanh_output ** 2)
