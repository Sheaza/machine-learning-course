import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


dataset = pd.read_csv('data/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

lin_regressor = LinearRegression()
lin_regressor.fit(x, y)
print(lin_regressor.predict([[6.5]]))

# transforming feature for poly regression
pol_feat = PolynomialFeatures(degree=4)
x_poly = pol_feat.fit_transform(x)

pol_regressor = LinearRegression()
pol_regressor.fit(x_poly, y)

test = [[6.5]]
test_poly = pol_feat.fit_transform(test)
print(pol_regressor.predict(test_poly))

# plot shows ideal function for polynomial regression usage
plt.scatter(x, y, c='red')

# comparison
plt.plot(x, lin_regressor.predict(x), c='blue', label='Linear')
plt.plot(x, pol_regressor.predict(x_poly), c='black', label='Polynomial')
plt.title("Linear and Polynomial comparison")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()