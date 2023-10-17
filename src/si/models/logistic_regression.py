import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset
from src.si.statistics.sigmoid_function import sigmoid_function
from src.si.metrics.accuracy import accuracy
from src.si.model_selection.split import stratified_train_test_split





class LogisticRegression:
    """
    The LogisticRegression is a logistic model using the L2 regularization.
    This model solves the logistic regression problem using an adapted Gradient Descent technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    alpha: float
        The learning rate
    max_iter: int
        The maximum number of iterations

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the logistic model.
        For example, sigmoid(x0 * theta[0] + x1 * theta[1] + ...)
    theta_zero: float
        The intercept of the logistic model
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

    def fit(self, dataset: Dataset) -> "LogisticRegression":
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LogisticRegression
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
            #aplicar a função sigmoid por ser regressao, onde sei que abaixo de 0.5 pertence a classe 0 e acima pertence a classe 1
            y_pred=sigmoid_function(y_pred)

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
        predictions=sigmoid_function(np.dot(X, self.theta) + self.theta_zero) #tenho de aplicar sigmoid as previsoes porque estao a trabalhar no regresao logistica

        # convert the predictions to 0 or 1 (binarization)
        return  np.where(predictions >= 0.5, 1, 0) #alterei do prof meti mais simples e acho que da

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
        return accuracy(dataset.y, y_pred)

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
        predictions=sigmoid_function(np.dot(X, self.theta) + self.theta_zero) #tenho de aplicar sigmoid as previsoes porque estao a trabalhar no regresao logistica
        cost = (dataset.y * np.log(predictions)) + (1 - dataset.y) * np.log(1 - predictions)
        cost = np.sum(cost) * (-1 / dataset.shape()[0])
        cost = cost + (self.l2_penalty * np.sum(self.theta ** 2) / (2 * dataset.shape()[0]))
        return cost
    
    if __name__ == '__main__':
    # import dataset

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = stratified_train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)

    print(model.theta)
    print(model.theta_zero)

    print(model.predict(dataset_test))

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")

    # plot the cost history
    import matplotlib.pyplot as plt

    plt.plot(list(model.cost_history.keys()), list(model.cost_history.values()))
    plt.show()
