import numpy as np

from src.si.data.dataset import Dataset

def read_data_file (filename:str, sep:str=None, label:bool=False) -> Dataset:
    """
    Have the capacity to read a data file into a dataset
    
    Arguments
    ----------
    filename: path to the file
    sep:separator used, by defaault put None
    label: the label collumn , by default is None
    """
    raw_data= np.genfromtxt(filename, delimiter=sep) #pega nos dados de um ficheito texto e converte numa matrix Numpy

    if label is not None:
        X=raw_data[:,:-1]
        y=raw_data[:,-1]

    else:
        X=raw_data
        y=None

    return Dataset(X,y) # onde com o codigo anterior defino X e y

def write_data_file (filename:str, dataset:Dataset, sep:str=None, label:bool=False) -> None:
    """
    Have the capacity to read a data file into a dataset
    
    Arguments
    ----------
    filename: path to the file
    dataset: the Dataset object
    sep: separator used, by defaault put None
    label: the label collumn , by default is None 
    """
    if not sep:
        sep=" " # alguma razao em especifico?

    if label is not None:
        data = np.hstack((dataset.X, dataset.y.reshape(-1, 1))) #dados vai ser x mais o vetor y tudo concatenado, com a particular do reshape do y pq é para assumir o memso numero de samples/subjects

    else:
        data=dataset.X

    return  np.savetxt(filename, data, delimiter=sep) #guarddar a matriz data num ficheiro com um delimator especifico (seprara por virgulas os valores neste caso)  
