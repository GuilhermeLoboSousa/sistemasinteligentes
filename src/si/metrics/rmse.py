import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
from src.si.data.dataset import Dataset

def rmse (y_true,y_pred) -> float:
    """
    It calculates the Root Mean Squared Error metric
    Args:
        y_true (np.ndarray): Real values
        y_pred (np.ndarray): Predicted values
    Returns:
        float: RMSE between real and predicted values
    """
    return np.sqrt((np.sum((y_true-y_pred)**2))/len(y_true))

if __name__ == '__main__':
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    print(rmse(y_true, y_pred)) #top