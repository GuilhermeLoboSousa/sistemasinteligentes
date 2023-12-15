import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

import copy
from abc import abstractmethod
from typing import Union
import numpy as np

from src.si.neural_networks.layers import Layer

class Dropout(Layer):#conjunti de valores do neuronio temporaraiamente desligado
    """
    A randomset of neurons is temporarily ignored (dropped out) during training, helping prevent overfitting by promoting robustness and generalization in the model.
    some neurons are off-this is some values of X are multiply by zero
    """
    def __init__(self, probability: float):
        """
        Initialize the dropout layer.

        Parameters
        ----------
        probability: float
            The dropout rate, between 0 and 1.
            probability to desconet some connections
        Attributes
        ----------
        mask: numpy.ndarray
            binomial mask that sets some inputs to 0 based on the probability
        input: numpy.ndarray
            the input to the layer
        output: numpy.ndarray
            the output of the layer

        
        """
        super().__init__() #nao percebo bem pq esta aqui- talvez pelos abstract
        self.probability = probability #se for 50 significa que naquela layer tenho 50 % neuronios ativos e 50 % inativos
        #estimated parameters
        self.input = None
        self.output = None
        self.mask=None #matrix of 0 and 1

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
        if training is True:#se tivermos a treinar fazemos o tal desligar(dropout)
            scaling_factor=1-(1-self.probability)#imaginem que temos [2,2],[2,2] total=2+2+2+2=8 ao desligar(50 % ficaria com ) [2,0],[0,2]=4 logo preciso do scaling factor para [4,0],[0,4]=8
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape) #dá algo com 0 e 1, por exemplo se a prob for 50 entao em 6 neuronios [0,0,1,1,0,1] 3 ligados e desligados
            self.output=input*self.mask*scaling_factor #estou a desligar alguns nos multiplicando por zero e os restante que nao foram desligados ainda sao multiplicado pelo scaling factor como mostrei antes
            return self.output
        else:#se nao tivermos em treino, for teste, ai já nao se faz dropout
            self.input = input
            return self.input

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: np.ndarraay
            The output error of the layer.

        Returns
        -------
        ndarray
            The output error of the layer.
        """
        return self.mask * output_error #basicamente pegamos em [0,1,0] 3 neuronios e multiplicamos pelo array de erro logo é esperado ter 2 neuronios desligadoe e portanto so faz back de 1 neuronio


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
        return 0 #nao há atualizações de pesos nem bias 

if __name__ == "__main__":

    dropout_layer = Dropout(probability=0.5)

    input_data = np.random.rand(3, 4)  # Assuming 3 samples with 4 features each

    output_data = dropout_layer.forward_propagation(input_data, training=True)
    print("Forward Propagation (Training):\n", output_data)

    output_error = np.random.rand(*output_data.shape)

    backward_output_error = dropout_layer.backward_propagation(output_error)
    print("\nBackward Propagation:\n", backward_output_error)
