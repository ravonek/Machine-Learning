import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


np.random.seed(1)
x = np.random.rand(100, 1)*10
y = x + 5 + np.random.randn(100,1)

model = linear_model.LinearRegression()
model.fit(x, y)

x_plot=np.linspace(0,11, 200)
x_plot = x_plot.reshape(-1, 1)
y_predicted = model.predict(x_plot)


plt.figure()
plt.scatter(x, y)
plt.plot(x_plot, y_predicted, color = "red")
plt.show()