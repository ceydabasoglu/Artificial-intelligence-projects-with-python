# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:55:17 2023

@author: pc
"""

"""
Multiple Linear Regression
"""

import pandas as pnds
import matplotlib.pyplot as pyplt
# sklearn library
from sklearn import linear_model


df = pnds.read_csv("multilinearregression.csv",sep = ";")
print(df)

# linear regression 

regression = linear_model.LinearRegression()
regression.fit(df[['area', 'numberOfRoom', 'buildingAge']], df['price'])

#predict

print("Area:250, number of room: 3, building age:15, Possible price of the house : ", regression.predict([[250,3,15]]))
print("Area:250, number of room: 6, building age:0, Possible price of the house : ",regression.predict([[250,6,0]]))
print("Area:165, number of room: 4, building age:25, Possible price of the house : ",regression.predict([[165,4,25]]))
print(regression.predict([[360,4,22]]))

#to calculate at once
price_list = [[250,3,15],[250,6,0],[165,4,25],[360,4,22]]
print(regression.predict(price_list))

print("coefficient :", regression.coef_)
print("intercept :", regression.intercept_)

# Multiple Linear regression formula
# y= a + b1X1 + b2X2 + b3X3 

a = regression.intercept_
b1 = regression.coef_[0]
b2 = regression.coef_[1]
b3 = regression.coef_[2]

x1 = 250
x2 = 3
x3 = 15
y = a + b1*x1 + b2*x2 + b3*x3
print(y)