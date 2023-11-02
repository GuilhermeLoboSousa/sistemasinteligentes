import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
from typing import Callable, Dict, List

import numpy as np
from src.si.data.dataset import Dataset


def k_fold_cross_validation(model, dataset: Dataset, scoring: callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds. #quanto maior mais tempo demora
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model on each fold.
    """
    num_samples = dataset.X.shape[0]
    fold_size = num_samples // cv #vamos querer quebrar o dataset em n foldes e que sejam todas no meusmo tamanho basicamente é pegar por exemplo na sample 1-5;6-10;11-15 etc
    scores = []

    # Create an array of indices to shuffle the data
    if seed is not None:
        np.random.seed(seed) #garantir reprodutibilidade na aleatoriedade
    indices = np.arange(num_samples) #criar uma matriz com os indices 
    np.random.shuffle(indices) #vai dar um shuffle nesses indices

    for fold in range(cv): #para cada cross validation
        # Determine the indices for the current fold
        #o que referi em cima com o exmplo 0 ao 5,5 ao 10 , 10 ao 15 ; atenção que em python o ultimo é "falso"  
        start = fold * fold_size
        end = (fold + 1) * fold_size

        # Split the data into training and testing sets
        test_indices = indices[start:end] #test ou seja tenho por exemplo lista de indices[2,23,25,26]e para o test fico com indices desta lista do 0 ao 3 ou seja as samples[2,23,25] sera o dataset teste
        train_indices = np.concatenate((indices[:start], indices[end:])) #restante é treino

        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices])
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices])

        # Fit the model on the training set and score it on the test set
        model.fit(dataset_train) #treino
        fold_score = scoring(dataset_test.y, model.predict(dataset_test)) if scoring is not None else model.score(
            dataset_test)#permitir avaliar por accuracy ou outra maneira de scoring, ou seja nao ficamos limitados ao que temos no metodo score dos modelos
        scores.append(fold_score) #para o test

    return scores

def leave_one_out_cross_validation(model, dataset: Dataset, scoring: callable = None, seed: int = None) -> List[float]:
    """
    Perform Leave-One-Out Cross-Validation (LOOCV) on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model for each data point.
    """
    num_samples = dataset.X.shape[0]
    scores = []

    # Create an array of indices to shuffle the data
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for i in range(num_samples):
        # Use data point i as the test set, and the rest as the training set
        train_indices = np.delete(indices, i)
        test_indices = [i]

        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices])
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices])

        # Fit the model on the training set and score it on the test set
        model.fit(dataset_train)
        fold_score = scoring(dataset_test.y, model.predict(dataset_test)) if scoring is not None else model.score(
            dataset_test)
        scores.append(fold_score)

    return scores

if __name__ == '__main__':
    # import dataset
    from src.si.data.dataset import Dataset
    from src.si.models.knn_classifier import KNNClassifier

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

    # initialize the KNN
    knn = KNNClassifier(k=3)

    # cross validate the model
    scores_ = k_fold_cross_validation(knn, dataset_, cv=5, seed=1)

    # print the scores
    print(scores_)
    # print mean score and standard deviation
    print(f'Mean score: {np.mean(scores_)} +/- {np.std(scores_)}')

    # LOOCV
    scores_ = leave_one_out_cross_validation(knn, dataset_, seed=1)

    # print the scores
    print(scores_)
    # print mean score and standard deviation
    print(f'Mean score: {np.mean(scores_)} +/- {np.std(scores_)}')