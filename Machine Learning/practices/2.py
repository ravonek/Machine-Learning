import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
dt = pd.read_csv("houses.csv")
dt.columns = ["living_space", "size_of_property", "price"]

# Данные
x = dt["living_space"].values
y = dt["price"].values

# Линейная регрессия: полином степени 1
degree = 3
coeffs = np.polyfit(x, y, degree)
func = np.poly1d(coeffs)

# Подготовка линии для графика
x_line = np.linspace(x.min(), x.max(), 200)
y_line = func(x_line)

y_pred = func(x)

area = 280
pred_price = func(area)
print(pred_price)


# График
plt.figure()
plt.scatter(x, y, color="red")
plt.plot(x_line, y_line, color="blue")

plt.show()
