import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, np.nan],
              [7, np.nan, 9]])
y = np.array([0, 1, 0])
features = ['feature_1', 'feature_2', 'feature_3']
label = 'target'

print(np.random.permutation(X.shape[0]))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_samples = 100
    num_features = 5
    X = np.random.rand(num_samples, num_features)  # Dados de recursos aleatórios
    y = np.random.randint(0, 2, num_samples)  # Rótulos aleatórios (assumindo uma classificação binária)


    dataset = Dataset(X=X,y=y)

    # Calcule a variância para diferentes valores de k
    variances = []
    for k in range(2, 100):  # Suponha que você deseja testar k de 1 a 10
        kmeans = Kmeans(k=k)
        kmeans.fit(dataset)
        labels = kmeans.labels
        unique_values, counts = np.unique(labels, return_counts=True)#quantas vezes cada valor da label aparece
        print(counts)
        total_samples = len(labels)
        percentages = (counts / total_samples) * 100
        variance = np.var(percentages)
        variances.append(variance)
        print(variances)

    # Plote a curva da variância em relação a k
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 100), variances, marker='o', linestyle='-', color='b')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Variancia das Percentagens')
    plt.title('Método do Cotovelo (Elbow Method)')
    plt.show()
    #perguntar ao prof