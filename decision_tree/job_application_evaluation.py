# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:48:40 2023

@author: pc
"""

#DECISION TREE CLASSIFICATION
#Artificial Intelligence Evaluation of Job Applications with Decision Trees
import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")
print(df)

mapping = {'Y' : 1, 'N' : 0}

df['Accepted for job'] = df['Accepted for job'].map(mapping)
df['Is it working now?'] = df['Is it working now?'].map(mapping)
df['Top10 University?'] = df['Top10 University?'].map(mapping)
df['did the internship with us?'] = df['did the internship with us?'].map(mapping)
mapping_education = {'BS' : 0, 'MS' : 1, 'PhD' : 2}
df['Education level'] = df['Education level'].map(mapping_education)
print(df)

y = df['Accepted for job'] 
x = df.drop(['Accepted for job'], axis = 1)
print(x)

classf = tree.DecisionTreeClassifier()
claasf = classf.fit(x,y)
 
#PREDICTION
print ("Is it accepted for a job :", classf.predict([[7, 1, 4, 0, 0, 0]])) 

print ("Is it accepted for a job :",claasf.predict([[3, 0, 6, 0, 1, 0]]))

print ("Is it accepted for a job :",claasf.predict([[3, 1, 6, 0, 0, 0]]))

print ("Is it accepted for a job :",classf.predict([[25, 0, 4, 1, 1, 1]]))
