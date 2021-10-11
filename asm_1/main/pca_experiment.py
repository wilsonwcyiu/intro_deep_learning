#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

import pandas as pd
from pandas import DataFrame

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])



from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
input_data_list = df.loc[:, features].values
# Separating out the target
output_y_list = df.loc[:, ['target']].values


# Standardizing the features
input_data_list = StandardScaler().fit_transform(input_data_list)

pca = PCA(n_components=2)
x_y_ndarray: np.ndarray = pca.fit_transform(input_data_list)

x_y_df: pd.DataFrame = pd.DataFrame(data = x_y_ndarray, columns = ['x component', 'y component'])

x_y_and_result_df: pd.DataFrame = pd.concat([x_y_df, df[['target']]], axis = 1)




fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('x component', fontsize = 15)
ax.set_ylabel('y component', fontsize = 15)
ax.set_title('2D PCA', fontsize = 20)

target_list: list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
color_list: list = ['r', 'g', 'b']
for target, color in zip(target_list, color_list):
    print(target, color)

    is_indices_to_keep: bool = x_y_and_result_df['target'] == target

    tmp_x = x_y_and_result_df.loc[is_indices_to_keep, 'x component']
    # print(tmp_x)
    # tmp_x = list(tmp_x)
    print(type(tmp_x))
    exit()


    ax.scatter(tmp_x
               , x_y_and_result_df.loc[is_indices_to_keep, 'y component']
               , c = color
               , s = 50)
ax.legend(target_list)
ax.grid()

print("test")
plt.show()