import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("data/Position_Salaries.csv")

x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# we don't need to scale features for decision tree
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])
print(y_pred)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, c="black")
plt.plot(x_grid, regressor.predict(x_grid), c='red')
plt.show()
