import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import pandas as pd

from src.si.data.dataset import Dataset

def read_csv (filename:str, sep:str=",", features:bool=None, label:bool=None )->Dataset:
    """
    Allow to read a csc file into a Dataset object
    
    Arguments
    --------------
    filename:str 
        the path tho the file
    sep:str 
        separator used in the file (most comumn is , or ;) 
    features: bool,optional 
        if the file has a header- default is false
    label: bool, optional
        if the file has a label y- default is false
    
    Returns
    -------
    Dataset
        The dataset object
    """
    data=pd.read_csv(filename,sep=sep)
    if features and label:
        features=data.columns[:-1] # nome de todas as coluans menos a ultim
        label=data.columns[-1] # nome da coluna label
        X=data.iloc[:,:-1].to_numpy() #todas as linhas e todas as colunas menos a ultima
        y=data.iloc[:,-1].to_numpy() #todas as linhas apenas da ultima coluna

    elif features and not label:
        features=data.columns
        X=data.to_numpy()
        y=None

    elif not features and label:
        features=None
        label=data.columns[-1] #eu penso que o mais correto seria isto pq eu sei que a ultima vai ser a label y- perguntar ao prof 
        X=data.iloc[:,:-1].to_numpy()
        y=data.iloc[:,-1].to_numpy()

    else:# sem features nem label
        X = data.to_numpy() # é tudo considerado X
        y = None #portanto nenhuma é y
        features = None # reforçar a propria condição
        label = None    

    return Dataset(X,y,features=features, label=label) #vai buscar a classe já criada, o que tivemos aqui a fazer foi preparar para ler o ficheiro consoante certas condições explicitas anteriormente

def write_csv (filename:str,dataset:Dataset, sep:str=",", features:bool=None, label:bool=None ):
    """
    Allow to transform Dataset object into a csv file
    
    Arguments
    --------------
    filename:str 
        the path tho the file
    sep:str, optional
      separator used in the file (most comumn is , or ;) 
    features:bool,optional 
        if the file has a header- default is false
    label:bool,optional 
        if the file has a label y- default is false
    """
    data=pd.DataFrame(dataset.X) # ir buscar a nossa matrix X como data

    if features is not None:
        data.columns=dataset.features # nome das colunas vvai ser obviamente o nome das features (que é uma lista de str)
    
    if label is not None:
        data[dataset.label]=dataset.y # valores da variável dataset.y para uma coluna específica do DataFrame queé a coluna da label
    
    data.to_csv(filename,sep=",", index=False) # sem retorno especifico apenas a fazer o que é pedido

