# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:42:42 2023

@author: pc
"""

#POLYNOMIAL LINEAR REGRESSION
#HR department salary scale calculation
#Polynomial Linear Regression Formula
#y = a + b1x + b2x^2 + b3x^3 + b4x^4 + ....... + bN*x^

import pandas as pds
import matplotlib.pyplot as pylt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Veri setimizi pandas yardımıyla alıp dataframe nesnemiz olan df'in içine aktarıyoruz..
df = pds.read_csv("polynomial.csv",sep = ";")
poly_regression = PolynomialFeatures(degree = 5)
print(df)

pylt.scatter(df['experience'],df['salary'])
pylt.xlabel('Experience (year)')
pylt.ylabel('Salary')
pylt.savefig('1.png', dpi=300)
pylt.show()



x_polynomial = poly_regression.fit_transform(df[['experience']])

regression = LinearRegression()
regression.fit(x_polynomial,df['salary'])

y_head = regression.predict(x_polynomial)
pylt.plot(df['experience'],y_head,color= "red",label = "polynomial regression")
pylt.legend()

#veri setimizi de noktlar olarak scatter edelim ve görelim uymuş mu polynomial regression:
pylt.scatter(df['experience'],df['salary'])   

pylt.show()

x_polynomial2 = poly_regression.fit_transform([[7.5]])
print(regression.predict(x_polynomial2))