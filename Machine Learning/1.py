import matplotlib.pyplot as plt
import numpy as np
import math

N = 50
x = np.linspace(0, 2 * math.pi, N)
y = [math.sin(i) for i in x]

fig = plt.figure()
plt.scatter(x, y)
fig.show()

# Цветовая карта
from matplotlib.colors import ListedColormap

vals = np.empty((N, 4))
vals[:, 0] = np.linspace(0, 131 / 255, N)       # Красный
vals[:, 1] = np.linspace(132 / 255, 184 / 255, N)  # Зелёный
vals[:, 2] = np.linspace(77 / 255, 26 / 255, N)    # Синий
vals[:, 3] = 1  # Прозрачность

my_map = ListedColormap(vals)

# Финальный график
fig = plt.figure()
plt.scatter(x, y,
            c=np.linspace(0, 1, N),
            cmap=my_map,
            marker='s',
            s=np.linspace(1, 500, N))
fig.show()
