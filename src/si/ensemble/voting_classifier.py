import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.
    It is like a democracy, for example 3 models predict[1,0,1] so the classe 1 is the winner

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.

    Attributes
    ----------
    """
    def __init__(self, models):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.

        """
        # parameters
        self.models = models #modelos que va ser utilizados

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : VotingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset) #treinar todos os modelos (que estao na lista) no mesmo dataset

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """

        # helper function
        def _get_majority_vote(pred: np.ndarray) -> int:#democracia
            """
            It returns the majority vote of the given predictions

            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of

            Returns
            -------
            majority_vote: int
                The majority vote of the given predictions
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            #[0,1]-labels [1,2]-counts-exemplo para primeira linha
            #labels[1(indice nas counts onde o numero é maior)] e portanto o label [1] vai ser a classe que aparece mais vezes nesta primeira linha era 1
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        #avaliar as previsoes feitas por cada modelo , sendo que obtemos algo como [[0,1,1]-modeloA [1,1,0]-modelo B [1,1,1]-modeloC]
        #mas nos queremos fazer democacria, ou seja escolher o que aparece mais por sample em todos os modelos daí a tranposta para ficar algo:
        # passamos a ter [[0,1,1]-modeloA,B,C [1,1,0]-modeloA,B,C [1,1,1]-modeloA,B,C]
        #agora podemos olhar para cada linha na matrix
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)#neste caso ficariamos com label 1-

    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    # import dataset
    from src.si.data.dataset import Dataset
    from src.si.model_selection.split import train_test_split
    from src.si.models.knn_classifier import KNNClassifier
    from src.si.models.logistic_regression import LogisticRegression
    from src.si.models.decision_tree_classifier import DecisionTreeClassifier

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
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)


    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))