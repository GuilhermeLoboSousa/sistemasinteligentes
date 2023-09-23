import numpy as np

# Seus dados da coluna y (exemplo)
y_data = np.array([[1, 0], [0, 1], [1, np.nan], [1, 0], [0, np.nan], [1, 0], [0, 1], [0, 1], [1, 0]])

novo=np.isnan(y_data).any(axis=1)
indice=np.where(novo)
indice
y_data[~novo]
y_data[novo]
nan_columns = np.isnan(y_data).any(axis=0) 
a=np.where(nan_columns)[0]
for x in a:
    print(a)
    col = y_data[:, a]
    np.isnan(col)
    col[np.isnan(col)]