import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
from abc import abstractmethod

import numpy as np


class Optimizer:#abstract metod sem init

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate #parametro que pode depois no modelo ser otimizado

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights. #melhorar, tenta acelarar a convergencia pq acumula gradientes passados
        """
        super().__init__(learning_rate) #vai buscar o parametro learning rate
        self.momentum = momentum
        self.retained_gradient = None #accumulated gradient from previous epochs(iteractions)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray #perda associada aos pesos
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:#se nao estiver inicializado fica uma matriz de zeros que depois com o back propagation sera atualizado
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient #lembrar que a aformula com temos t-1 significa que estamos nesse ponto e estamos a tentar passar para o seguinte

class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.01, beta_1:float=0.9,beta_2:float=0.999,epsilon: float = 1e-8):
        """
        Initialize the optimizer.
        Combination of RMSprop and SGD with momentum

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1:float
            The exponential decay rate for the 1st moment estimates
        beta_2:float
            The exponential decay rate for the 2nd moment estimates
        epsilon:float
            A small constantfor numerical stability
        """
        super().__init__(learning_rate) #vai buscar o parametro learning rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        #estimated parameters
        self.m=None #moving average m
        self.v=None #moving average v
        self.t=0 #time stamp (epoch), initialized as 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray #perda associada aos pesos
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.m is None and self.v is None:#se nao estiver inicializado fica uma matriz de zeros que depois com o back propagation sera atualizado
            self.m = np.zeros(np.shape(w))
            self.v= np.zeros(np.shape(w))
        else:
            self.t +=1
            
            self.m=self.beta_1*self.m + ((1-self.beta_1)*grad_loss_w)
            self.v=self.beta_2*self.v + ((1-self.beta_2)*(grad_loss_w**2))
            #correção bias
            m_hat=self.m/(1-self.beta_1**self.t)#confirmar se é mesmo ao quadrado
            v_hat=self.v/(1-self.beta_2**self.t)

            w= w - (self.learning_rate*(m_hat/(np.sqrt(v_hat) + self.epsilon)))
        
        return w 
        