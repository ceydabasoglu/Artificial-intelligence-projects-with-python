# -*- coding: utf-8 -*-
"""
Ceyda Başoğlu
"""

#KNN (K- Nearest Neighbours)
#Project to predict diabetes using dataset

import pandas as pnds
import matplotlib.pyplot as pyplt
import numpy as nmpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Outcome = 1 Diabetic
# Outcome = 0 Healthy
data = pnds.read_csv("diabetes.csv")
print(data.head())

diabetic = data[data.Outcome == 1]
healthy = data[data.Outcome == 0]

#I make a sample drawing according to the glucose value
#At the end of the project, the machine learning model will make predictions based on all data, not just glucose.

pyplt.scatter(healthy.Age, healthy.Glucose, color="green", label = "Healthy people", alpha = 0.4)
pyplt.scatter(diabetic.Age, diabetic.Glucose, color= "red", label = "Diabetic people", alpha = 0.4)
pyplt.xlabel("Age")
pyplt.ylabel("Glucose")
pyplt.legend()
pyplt.show()

y = data.Outcome.values
x_raw_data = data.drop(["Outcome"],axis=1)   

#we normalization we update all values in x_raw_data so that they are only between 0 and 1
# If we don't normalization like this, high numbers will overwhelm the small numbers and may confuse the KNN algorithm!

x = (x_raw_data - nmpy.min(x_raw_data))/(nmpy.max(x_raw_data)-nmpy.min(x_raw_data))

print("Raw data before normalization:\n")
print(x_raw_data)

print("\n\n\nThe data we will give to artificial intelligence for training after normalization:\n")
print(x)

# we separate our train data and test data
# Our train data will be used to learn the system to distinguish between a healthy person and a sick person
# our test data will be used to test whether our machine learning model can correctly distinguish between sick and healthy people...

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("Confirmation test result of our test data for K=5 ", knn.score(x_test, y_test))

# let's determine the best k value..
count = 1
for k in range(1,15):
    knn_new = KNeighborsClassifier(n_neighbors = k)
    knn_new.fit(x_train,y_train)
    print(count, "  ", "Accuracy rate: %", knn_new.score(x_test,y_test)*100)
    count += 1
    
# For a new patient estimate:
from sklearn.preprocessing import MinMaxScaler

# normalization
sc = MinMaxScaler()
sc.fit_transform(x_raw_data)

new_prediction = knn.predict(sc.transform(nmpy.array([[6,148,72,35,0,33.6,0.627,50]])))
print(new_prediction[0])    