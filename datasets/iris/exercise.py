import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
import pandas as pd
from src.si.data.dataset import Dataset
from src.io.csv_file import read_csv, write_csv 
from src.io.data_file import *
from src.si.statistics.f_classification import f_classification
from src.si.feature_selection.percentile import Percentile

filename = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\iris\iris.csv"

#1
dataset=read_csv(filename, sep=",")
dataset

#2
penultimo=dataset.X[:,-2]
dataset.features[-2] #esta a ir buscar um nome aleatorio deveria de ir buscar petal_width
penultimo.shape

#3
last_10=dataset.X[-10:,:-1]
last_10

#b
mean_per_feature = np.mean(last_10, axis=0)
mean_per_feature
#tentaar indicar a media por cada 
features_names=dataset.features[:-1]
mean_dict={features_names[i]:mean_per_feature[i] for i in range(len(features_names))}
mean_dict

#4
dados = dataset.X[:,:-1]
condicao = (dados <= 6).all(axis=1) # para ir buscar apenas [] e poder fazer o resto, caso contrario aparece [[]]
condicao
soma = condicao.sum()
print(soma)

#5
dataset.X
novo=dataset.X[:,-1]
np.sum(novo=="Iris-setosa")

#exercicio aula 22


iris=read_csv(filename, sep=",",features=True,label=True)
labels = dataset.label
print(labels)
#percentile_selector = Percentile(score_func=f_classification, percentile=25)  
#percentile_selector.fit(dataset)
#dataset_filtered = percentile_selector.transform(dataset)
#print("Features after percentile selection:")
#print(dataset_filtered.features)
