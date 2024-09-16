
import numpy as np
from sklearn import  datasets, linear_model

x1 = np.array([[60, 40, 100]]).T
x2 = np.array([[2, 2, 3]]).T
x3 = np.array([[10, 5, 7]]).T

X =  np.hstack((x1, x2, x3))
Y = np.array([10, 12, 20])

a = np.dot(X.T, X)
b = np.dot(X.T, Y)
w = np.dot(np.linalg.pinv(a), b)

w1, w2, w3 = w[0], w[1], w[2]
print(w1, w2, w3)

y1 = w1*50 + w2*2 + w3*8
print("Gia nha: ", y1)

print("//////////////////////////////////////")

lr = linear_model.LinearRegression(fit_intercept=False)
lr.fit(X, Y)

w1, w2, w3 = lr.coef_

print(w1, w2, w3)
print("Gia nha: ", y1)