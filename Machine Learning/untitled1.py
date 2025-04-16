import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd

np.random.seed(1)
x = np.random.rand(100, 1)*3
z = 40 + x + 2*(x**2) + np.random.randn(100,1)*5

data = pd.DataFrame(x)
