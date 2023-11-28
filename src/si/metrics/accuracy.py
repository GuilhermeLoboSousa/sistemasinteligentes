import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset

def accuracy(y_true:np.ndarray,y_pred:np.ndarray) -> float:
    """
    It returns the accuracy of the model on the given dataset
    (TN + TP)/ (TN +TP + FN +FP)

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    accuracy: float
        The accuracy of the model
    """
    def correct_format(y):
        if len(y[0]) == 1:
            corrected_y = [np.round(y[i][0]) for i in range(len(y))]
        else:
            corrected_y = [np.argmax(y[i]) for i in range(len(y))]
        return np.array(corrected_y)
    if isinstance(y_true[0], list) or isinstance(y_true[0], np.ndarray):
        y_true = correct_format(y_true)
    if isinstance(y_pred[0], list) or isinstance(y_pred[0], np.ndarray):
        y_pred = correct_format(y_pred)
    return np.sum(y_pred == y_true) / len(y_true)
     #o y_pred vai ser algo que vai de acordo ou nao ao que esta no dataset (y_true), logo ao fazer == verificamos onde se verifica true negative as positve
#esse == dá um array de boleanos que podemos contar fazendo o np.sum
# e depois dividimos por o len de y_tru ou de y_pred pois quer um ou outro vai ter todas as possibilidades quer seja TN,TP,FN,FP
#indo por isso ao encontro da formula