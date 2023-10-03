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

#Exercicio Aula 1
#1
dataset=read_csv(filename, sep=",",features=True,label=True)

#2
penultimo=dataset.X[:,-2]
print(penultimo.shape)

#3
last_10=dataset.X[-10:,:]
print(last_10)

#b
mean_per_feature = np.mean(last_10, axis=0)
mean_per_feature
#tentaar indicar a media por cada 
features_names=dataset.features[:]
mean_dict={features_names[i]:mean_per_feature[i] for i in range(len(features_names))}
print(mean_dict)

#4
dados = dataset.X[:,:]
condicao = (dados <= 6).all(axis=1) # para ir buscar apenas [] e poder fazer o resto, caso contrario aparece [[]]
soma = condicao.sum()
print(soma)

#5
novo=dataset.y
total=np.sum(novo!="Iris-setosa")
print(total)

#exercicio aula 22


# iris=read_csv(filename, sep=",",features=True,label=True)
# percentile_selector = Percentile(score_func=f_classification, percentile=)  
# percentile_selector.fit(iris)
# dataset_filtered = percentile_selector.transform(iris)
# print("Features after percentile selection:")
# print(dataset_filtered.features)
