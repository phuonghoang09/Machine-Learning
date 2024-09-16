import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

#khai báo training dataset
height = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
width = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

#mở rộng ma trận data
one = np.ones((height.shape[0], 1))
heightBar = np.column_stack((one, height))

#thực hiện tính theo công tính linear regression
a = np.dot(heightBar.T, heightBar)
b = np.dot(heightBar.T, width)
w = np.dot(np.linalg.pinv(a), b)

w0, w1 = w[0], w[1]
print(w0, w1)

y1 = w1*155 + w0
y2 = w1*160 + w0
print("Input 155cm, true output 52kg, predicted output %.2fkg" %(y1))
print("Input 160cm, true output 56kg, predicted output %.2fkg" %(y2))

fig, ax = plt.subplots(figsize= (10, 5))
ax.scatter(height, width, color = "red")
ax.set(xlabel = "Height",
      ylabel = "Width")
ax.plot(height, w1*height + w0)
plt.show()

chieu_cao = int(input("Nhap chieu cao: "))
y1 = w1*chieu_cao + w0
print("Can nang = ", y1)

lr = linear_model.LinearRegression()
lr.fit(height, width)

w1, w0 = lr.coef_[0], lr.intercept_
print(w0, w1)