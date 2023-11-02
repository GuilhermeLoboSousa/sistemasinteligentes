import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import itertools
from typing import Callable, Tuple, Dict, Any

import numpy as np

from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import k_fold_cross_validation


def grid_search_cv(model,
                   dataset: Dataset,
                   hyperparameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 5) -> Dict[str, Any]:
    """
    Performs a grid search cross validation on a model.

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

    Returns
    -------
    results: Dict[str, Any]
        The results of the grid search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")#verificar se tenhoo hiperparametro que quero analisar num determinado modelo, ou seja nao posso ter k para o modelo do nb

    results = {'scores': [], 'hyperparameters': []} 

    # for each combination
    for combination in itertools.product(*hyperparameter_grid.values()):#permite fazer comninações entre parametros e respetivos valors
#combination seria algo para 'l2_penalty', alpha e max_iter do genero (1,0.001,1000) (1,0.001,2000) etc

        # parameter configuration
        parameters = {}#conseguimos acompanhar os paramentro 

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            #vou ficar com parameter-value, usando o exemplo de cima
            #l2_penalty - 1,alpha-0.001, max_iter-1000 etc
            setattr(model, parameter, value)#é o que permite definir estes novos atributos para o modelo em questão
            parameters[parameter] = value #dicionnario que guarda aquilo que foi trabalhado

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
    results_ = grid_search_cv(knn,
                              dataset_,
                              hyperparameter_grid=parameter_grid_,
                              cv=3)

    # print the results
    print(results_)

    # get the best hyperparameters
    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}")

    # get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")