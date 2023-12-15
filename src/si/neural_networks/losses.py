import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")

from abc import abstractmethod

import numpy as np


class LossFunction: #nao precisamos init- abstract metods

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction, based on the tru and predicted labels

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        #apenas seguir formula
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15) #limitar valores do y_pred, para nao andar com valores de infinito e assim
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)
    
class CategoricalCrossEntropy(LossFunction):#multiclass
    """
    Categorical Cross Entropy loss function.
    Measures the dissimilarity between predicted class probabilities and true one-hot encoded class labels
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15) #limitar valores do y_pred, para nao andar com valores de infinito e assim
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p)

if __name__ == '__main__':

    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2]])

    mse_loss = MeanSquaredError()
    mse_result = mse_loss.loss(y_true, y_pred)
    mse_derivative = mse_loss.derivative(y_true, y_pred)

    print(f"Mean Squared Error Loss: {mse_result}")
    print(f"Mean Squared Error Derivative: {mse_derivative}")

    bce_loss = BinaryCrossEntropy()
    bce_result = bce_loss.loss(y_true, y_pred)
    bce_derivative = bce_loss.derivative(y_true, y_pred)

    print(f"Binary Cross Entropy Loss: {bce_result}")
    print(f"Binary Cross Entropy Derivative: {bce_derivative}")

    cce_loss = CategoricalCrossEntropy()
    cce_result = cce_loss.loss(y_true, y_pred)
    cce_derivative = cce_loss.derivative(y_true, y_pred)

    print(f"Categorical Cross Entropy Loss: {cce_result}")
    print(f"Categorical Cross Entropy Derivative: {cce_derivative}") 
       