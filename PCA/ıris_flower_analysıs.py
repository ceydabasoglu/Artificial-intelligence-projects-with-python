# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:05:09 2023

@author: pc
"""

#PRINCIPAL COMPONENT ANALYSIS PCA
#iris flower analysis
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "pca_iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
print(df)

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x_features = df[features]

y_target = df[['target']]
x_features = StandardScaler().fit_transform(x_features)
print(x_features)
#PCA Projection 4 dimension to 2 dimension
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_features)
principalDf = pd.DataFrame(data = principalComponents, columns = ['component 1', 'component 2'])
print(principalDf)


final_dataframe = pd.concat([principalDf, df[['target']]], axis = 1)
print(final_dataframe)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['green', 'blue', 'red']

plt.xlabel('component 1')
plt.ylabel('component 2')

for target, col in zip(targets,colors):
    dftemp = final_dataframe[df.target==target]
    plt.scatter(dftemp['component 1'], dftemp['component 2'], color=col)
    
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())