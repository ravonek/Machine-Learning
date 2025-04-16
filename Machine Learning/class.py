import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import math

# Кол-во точек
n_points = 50

# Значения x и y (синус)
x = np.linspace(0, 2 * math.pi, n_points)
y = np.sin(x)

# Размеры точек от 10 до 200 (разница в 20 раз)
sizes = np.linspace(10, 200, n_points)

# СОЗДАЁМ СОБСТВЕННУЮ ЦВЕТОВУЮ КАРТУ (от тёмного зелёного к светлому)
vals = np.zeros((n_points, 4))  # (n, 4) для RGBA
# Цвет от RGB(0,132,77) → RGB(131,184,26)
vals[:, 0] = np.linspace(0, 131/255, n_points)   # R
vals[:, 1] = np.linspace(132/255, 184/255, n_points)  # G
vals[:, 2] = np.linspace(77/255, 26/255, n_points)    # B
vals[:, 3] = 1.0  #

# Создаем карту
my_map = ListedColormap(vals)

# Рисуем
fig, ax = plt.subplots()

# Используем c=x для сопоставления цветов и cmap
sc = ax.scatter(x, y,
                c=np.arange(n_points),  # значения от 0 до 49
                cmap=my_map,            # наша цветовая карта
                s=sizes,                # размеры
                marker='s',             # квадрат
                edgecolors='none')      # без рамки

# Настройки
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
