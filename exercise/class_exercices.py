import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
import pandas as pd
from src.si.data.dataset import Dataset
from src.io.csv_file import read_csv, write_csv 
from src.io.data_file import *
from src.si.statistics.f_classification import f_classification
from src.si.feature_selection.percentile import Percentile
from src.si.decomposition.pca import PCA
from src.si.statistics.euclidean_distance import euclidean_distance
from src.si.clustering.kmeans import Kmeans
from src.si.metrics.accuracy import accuracy
from src.si.metrics.rmse import rmse
from src.si.model_selection import *
from src.si.model_selection.split import train_test_split
from src.si.model_selection.split import stratified_train_test_split
from collections import Counter
from src.si.models.knn_regressor import KNNRegressor
from src.si.models.categorical_nb import CategoricalNB
from src.si.models.knn_classifier import KNNClassifier
from src.si.models.ridge_regression import RidgeRegression
from src.si.models.logistic_regression import LogisticRegression
from src.si.model_selection.cross_validate import k_fold_cross_validation
from src.si.model_selection.grid_search import grid_search_cv
from src.si.model_selection.randomized_search_cv import randomized_search_cv


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

#exercicio aula 2
iris=read_csv(filename, sep=",",features=True,label=True)
percentile_selector = Percentile(score_func=f_classification, percentile=25)  
percentile_selector.fit(iris)
dataset_filtered = percentile_selector.transform(iris)
print("Features after percentile selection:")
print(dataset_filtered.features)

#exercicio aula 3
iris=read_csv(filename, sep=",",features=True,label=True)
pca = PCA(n_components=3)
transformed_data = pca.fit_transform(iris)
print("Conjunto de Dados Transformado:")
print(transformed_data) #onde foi "projeto ao longo de pc1 -primeira colunca e ao longo de pc2-segunda coluna"
#como nao percebia muito do ouput obtido decidi adicionar o metodo para ver a representabilidade
print(pca.plot_variance_explained())

#exercicio aula 4
iris=read_csv(filename, sep=",",features=True,label=True) #iris tem 150 linhas por 4 colunas
train_data, test_data = stratified_train_test_split(iris, test_size=0.2, random_state=42)
print("Tamanho do treino:", train_data.shape())# 150 *0.8=120
# Verifique a proporção das classes nos conjuntos de treinamento e teste
label_counts = Counter(train_data.y)
labels_count_original=Counter(iris.y)
print(label_counts,labels_count_original) #mesma proproção parece correto

#exercicio aula 4
filename_cpu = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\cpu\cpu.csv"
cpu=read_csv(filename_cpu, sep=",",features=True,label=True) #iris tem 150 linhas por 4 colunas
train_data, test_data = stratified_train_test_split(cpu, test_size=0.2, random_state=42)
knn_regressor = KNNRegressor(k=3)  
knn_regressor.fit(train_data)
score = knn_regressor.score(test_data)
print(f'The rmse of the model is: {score}')

#exercicio aula 5
filename_cpu = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\cpu\cpu.csv"
cpu=read_csv(filename_cpu, sep=",",features=True,label=True)
train_data, test_data = stratified_train_test_split(cpu, test_size=0.20, random_state=42)
model = RidgeRegression()
model.fit(train_data)
score = model.score(test_data)
print(f"Score: {score}")
cost = model.cost(test_data)
print(f"Cost: {cost}")

#exercicio aula 5
filename_breast = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\breast_bin\breast-bin.csv"
breast=read_csv(filename_breast, sep=",",features=True,label=True)
train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)
model = LogisticRegression()
model.fit(train_data)
score = model.score(test_data)
print(f"Score: {score}")
cost = model.cost(test_data)
print(f"Cost: {cost}")

#exercicio ala 7 crossvalidation
filename_breast = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\breast_bin\breast-bin.csv"
breast=read_csv(filename_breast, sep=",",features=True,label=True)
model = LogisticRegression()
# cross validate the model
scores_ = k_fold_cross_validation(model, breast, cv=5, seed=1)
print(scores_)
print(f'Mean score: {np.mean(scores_)} +/- {np.std(scores_)}')

#exercicio aula 7 grid
filename_breast = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\breast_bin\breast-bin.csv"
breast=read_csv(filename_breast, sep=",",features=True,label=True)
model = LogisticRegression()
parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }
results_ = grid_search_cv(model,
                          breast,
                          hyperparameter_grid=parameter_grid_,
                          cv=3)

print(results_)
#scores
print(results_["scores"])
# get the best score
best_score = results_['best_score']
print(f"Best score: {best_score}")

#exercicio aula 7 random
filename_breast = r"C:\Users\guilh\OneDrive\Documentos\GitHub\sistemasinteligentes\datasets\breast_bin\breast-bin.csv"
breast=read_csv(filename_breast, sep=",",features=True,label=True)
model = LogisticRegression()
parameter_grid_ = {
        'l2_penalty':np.linspace(1, 10, 10),
        'alpha':  np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }
results_ = randomized_search_cv(model,
                          breast,
                          hyperparameter_grid=parameter_grid_,
                          cv=3,
                          n_iter=100)

print(results_)
#scores
print(results_["scores"])
# get the best score
best_score = results_['best_score']
print(f"Best score: {best_score}")
#no exercicio da aula 7 do grid o numero maximo de combinações é 8 entao
#cheguei a fazer com o  n_iter=8  para verificar se dava o mesmo resultado que em grid search e deu, o que oprova estar bem implementado
#deu isto :{'l2_penalty': 1, 'alpha': 0.0001, 'max_iter': 1000}, 'best_score': 0.9683908045977011}