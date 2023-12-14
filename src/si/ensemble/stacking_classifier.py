import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np

from src.si.data.dataset import Dataset
from src.si.metrics.accuracy import accuracy

class StackingClassifier:#primeiro treinamos com o dataset original um numero de modelos;2- as previsoes obtidas irao servir de treino para um modelo final, ou seja , o datset final tera como y as previsoes feitas pelos outro modelos
    """
    Train a final model, which uses as a training dataset the predictions of models trained with the original dataset
    Uses a set of models to predict the outcome. These predictions are used to train a final model which is then used to
    predict the final outcome of the output variable (Y).

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.
    fina_model:str 
        the model to make the final predictions

    Attributes
    ----------
    """
    def __init__(self, models:list[object], final_model:object):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.

        """
        # parameters
        self.models = models #modelos que va ser utilizados
        self.final_model=final_model#modelo final que vai fazer a previsao

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset) #treinar todos os modelos (que estao na lista) no mesmo dataset
        
        predictions=[]
        for model in self.models:
            prever=model.predict(dataset)
            predictions.append(prever)#fico com algo [[previsao modelo a],[previsao modelo b],etc]
        

        #ou seja o meu dataset treino vai agora ser precistions
        #fico agora com um array onde as linhas sao modelos e as colunas os valores de y para cada sample
        #logo terei de aplicar transposta para ficar com valores de  y para cadaa sample nas linhas e os modelos nas colunas
        predictions=np.array(predictions).T #tive de transformar em np array pq estava a dar erro
        self.final_model.fit(Dataset(dataset.X, predictions)) #faz um fit em bloco digamos assim datasetx LABEL MODELO 1, DATASETX LABEL MODELO 3 ,ETC
        return self
    
    def predict(self, dataset: Dataset) -> np.array:
        """Collects the predictions of all the models and computes the final prediction of the final model returning it.
        Args:
            dataset (Dataset): Dataset 
        Returns:
            np.array: Final model prediction
        """
        # mesma logica que aplicada no fit
        predictions = []
        for model in self.models:
            prever=model.predict(dataset)
            predictions.append(prever)
        
        predictions=np.array(predictions).T
        y_pred_final=self.final_model.predict(Dataset(dataset.X, predictions)) #em vez de ser o tipico label y de 1 coluna por x linhas, vamos ter tambem n colunas por x linhas (isto estava a causar um pouco de confusao), mas é possivel            
        return y_pred_final

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
    from src.io.csv_file import read_csv
    from src.si.model_selection.split import stratified_train_test_split
    from src.si.models.knn_classifier import KNNClassifier
    from src.si.models.logistic_regression import LogisticRegression
    from src.si.models.decision_tree_classifier import DecisionTreeClassifier

    filename_breast = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\breast_bin\breast-bin.csv"
    breast=read_csv(filename_breast, sep=",",features=True,label=True)
    train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)

    #knnregressor
    knn = KNNClassifier(k=3)
    
    #logistic regression
    LG=LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    #decisiontreee
    DT=DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')

    #final model
    final_modelo=KNNClassifier(k=3)
    modelos=[knn,LG,DT]
    exercise=StackingClassifier(modelos,final_modelo)
    exercise.fit(train_data)
    print(exercise.score(test_data))



