import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
from src.si.metrics.mse import mse


class RidgeRegression:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Gradient Descent technique (we want the lower derivate)
    try to avoid overfitting in linae regression models
    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations
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

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 10000,
                 patience: int = 5, scale: bool = True):
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
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegression
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

        # initialize the model parameters
        self.theta = np.zeros(n) #matriz de zeros, que tem tantos zeros como o numero de features dai ser o n
        #como o slide diz o theta sao coeficientes para cada feature
        self.theta_zero = 0 #apenas um valor que começa a zero

        i = 0 #inicio das iterações
        early_stopping = 0 # 
        # gradient descent-queremos sempre que va descendo 
        while i < self.max_iter and early_stopping < self.patience: #a função esta limitada ao numero de iterações e se os resultados obtidos forem piores que os anteriores, ou seja se a função custo do ciclo seguinte for maior ou igual à do cicloa anterior 
            # predicted y
            y_pred = np.dot(X, self.theta) + self.theta_zero #tetazero + teta1 *x1 + teta2*x2 etc

            # computing and updating the gradient with the learning rate~
            #seguir a formula do professor
            gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X) #alfa nao pode ser muito grande pq podemos passar o ponto otimo do gradiente, mas tb nao pode ser muito pequeno pq nao saimos do sitio
            #  penalty term
            penalization_term = self.theta * (1 - self.alpha * (self.l2_penalty / m)) #seguir  formula

            # updating the model parameters
            self.theta = ( penalization_term) - gradient #acho que esta mal nao deveria ser multiplicar???? porque o penalization term ja tem tudo
            #tirei o multiplixar e a verdade é que aproximou os valores da minha classe com o do sckitlearn
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y) # o x0 seria 1 por isso nao se coloca

            # compute the cost
            self.cost_history[i] = self.cost(dataset) #criar um dicionario de iteação-custo, onde a função cuto é auxiliar
            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]: #se a fubção custo deste ciclo for superior ou  à do anterior adicionamos ao erly_stoping e isso so pode acontecer 5 vezes -patience
                early_stopping += 1
            else:
                early_stopping = 0
            i += 1 #sempre a incrementar 1 no contador até max iterações

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
        return np.dot(X, self.theta) + self.theta_zero #seguir formula

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

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y)) #len do dataset.y é o numero de samples é o m


if __name__ == '__main__':
    # import dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3 # bombinação linear das caracteristicas em X mais um termo- objetivo é ter relação linear x,y
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegression()
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    new_data = Dataset(X=np.array([[3, 5]]),y=None) # X com "duas features " e vou rpever quanto vale y 

    y_pred_ = model.predict(new_data)#novo conjunto de dados é criado para tentar prever
    print(f"Predictions: {y_pred_}")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# make a linear dataset
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
dataset_ = Dataset(X=X, y=y)

# fit the model
model = Ridge(alpha=1.0)  # alpha é o equivalente ao l2_penalty
model.fit(X, y)  # Nesse caso, não é necessário criar um objeto Dataset

# get coefficients
print(f"Coefficients: {model.coef_}")


# compute the cost (MSE)
y_pred = model.predict(X)
cost = mean_squared_error(y, y_pred)
print(f"Mean Squared Error (MSE): {cost}")

# predict
new_data = np.array([[3, 5]])
y_pred = model.predict(new_data)
print(f"Predictions: {y_pred}")

#peprguntar ao professor porque dá ligeiramente diferente do sckit leanr?
#se aumentar o numero de iterações e a patient aproxima mais 
