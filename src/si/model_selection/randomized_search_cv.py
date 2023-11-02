import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import itertools
from typing import Callable, Tuple, Dict, Any

import numpy as np

from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv (model,dataset:Dataset, hyperparameter_grid: Dict[str, Tuple],scoring: Callable = None,cv: int = 5,n_iter:int=None):
    """
    Implements a parameter optimization strategy with cross validation using a number of random combinations selected from a distribution possible hyperparameters.
    more efficient and useful in large dataset
    makes n random combinations with the hyperparameters and may not give the optimal solution, but it gives a good combination in less time

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.
    n_iter:int
        number of hyperparameter random combinations to test

    Returns
    -------
    results: Dict[str, Any]
        The results of the grid search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    for parameter in hyperparameter_grid:
            if not hasattr(model, parameter):
                raise AttributeError(f"Model {model} does not have parameter {parameter}.")#verificar se tenhoo hiperparametro que quero analisar num determinado modelo, ou seja nao posso ter k para o modelo do nb
            
    results = {'scores': [], 'hyperparameters': []}
    for x in range(n_iter):#apenas quero que fça n_iter combinações
        parameters={}
        for chave, valores in hyperparameter_grid.items():#estou a "separar" as chaves dos possiveis valores, para depois conseguir escolher random valores
            valores_random=np.random.choice(valores)#escolher random os valores por cada chave ou seja teria por exemplo chave a valores 1,2,3 e estaria a escolher random 1 ou 2 ou 3
            parameters[chave]=valores_random #guardava a escolha feita
            setattr(model, chave, valores_random) #defenia as escolhas feita para serem argumentos da função

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score)) #estou a fazer o modelo com todas as combinações de hiperparamteros guardadas anteriormente e a guardar o score

        # add the hyperparameters
        results['hyperparameters'].append(parameters) #associar score ao hiperparametros e respetivos valores analisados

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])] #quero os hiperparametros que obtiveram maior score
    results['best_score'] = np.max(results['scores']) # e ja agora qual foi o valor desse score
    return results

if __name__ == '__main__':
    # import dataset
    from src.si.models.logistic_regression import LogisticRegression

    num_samples = 600
    num_features = 100
    num_classes = 2

    # random data
    X = np.random.rand(num_samples, num_features)  
    y = np.random.randint(0, num_classes, size=num_samples)  # classe aleatórios

    dataset_ = Dataset(X=X, y=y)

    #  features and class name
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "class_label"

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    results_ = randomized_search_cv(knn,
                              dataset_,
                              hyperparameter_grid=parameter_grid_,
                              cv=3,
                              n_iter=8)

    # print the results
    print(results_)

    # get the best hyperparameters
    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}")

    # get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")

    #fiz com o mesmo numero de combinações que utilizei no gridsearch e ambos score andam perto do 0.5