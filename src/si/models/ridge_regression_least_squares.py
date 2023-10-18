import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
from src.si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique (we want the lower derivate)
    try to avoid overfitting in linae regression models
    an optimal solution is guaranteed
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """

    def __init__(self, l2_penalty: float = 1,  scale: bool = True):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        patience: int
            The number of iterations without improvement before stopping the training
        scale: bool
            Whether to scale the dataset or not
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        if self.scale:#caso seja para fazer o scale vamos descobrir o x´ apresentado no slide
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X #caso contrario mantemos o dataset
        
        m, n = dataset.shape()
        X = np.c_[np.ones(m), X] #acrecentar uma coluna de 1 na primeira coluna
        penalty_matrix=self.l2_penalty*np.eye(n+1) # matriz identidade tera 0 e 1 na diagonal, sendo do tamanho fas features e como eu acrecentei uma feature no dataset coloquei o +1
        penalty_matrix[0,0]=0 # primeira posição da matriz é 0 para garantir que tetazero nao é penalizado
        print(penalty_matrix)
        #passo 5-aplicar formula
        transposed_X = X.T

        # Calcule a primeira parte da expressão analítica
        first_part = np.linalg.inv(transposed_X.dot(X) + penalty_matrix)

        # Calcule a segunda parte da expressão analítica
        second_part = transposed_X.dot(dataset.y)

        # Calcule os parâmetros do modelo
        thetas=first_part.dot(second_part)
        self.theta_zero=thetas[0] #indicaçõe sdo slide no passo 5 
        self.theta=thetas[1:] #theta (remaining elements)


        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X #fazer isto
        m, n = dataset.shape()
        X = np.c_[np.ones(m), X]
        previsao = X.dot(np.r_[self.theta_zero, self.theta])  # Concatena theta_zero e theta e realiza a multiplicação de matriz

        return previsao

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)



if __name__ == '__main__':
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([2, 3, 4, 5])
    dataset = Dataset(X, y)

    ridge_regression = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)

    ridge_regression.fit(dataset)

    y_pred = ridge_regression.predict(dataset)

    mse_result = mse(y, y_pred)

    print("Previsões minha class:", y_pred)
    print("MSE minha class:", mse_result)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge_regression = Ridge(alpha=0.1)

ridge_regression.fit(X, y)

y_pred = ridge_regression.predict(X)

mse_result = mean_squared_error(y, y_pred)

print("Previsões do sklearn:", y_pred)
print("MSE do sklearn:", mse_result)

#dá praticamente a mesma coisa!!
