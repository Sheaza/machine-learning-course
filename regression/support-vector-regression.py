import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('data/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# feature scaling must be applied after data split
# we don't scale binary features because they are already in the right range
# we don't apply feature scaling on dummy variables from one hot encoding
# dependent and independent variables must be in the same range

# to visualise prediction we will have to invert feature scaling

# reshape y to be 2d array

y = np.reshape(y, (len(y), 1))

# scaling (sc scale values somewhere between -(-3, 3))

xsc = StandardScaler()
ysc = StandardScaler()
x = xsc.fit_transform(x)
y = ysc.fit_transform(y)


# train

svr_reg = SVR(kernel="rbf")
svr_reg.fit(x, y)

pred = ysc.inverse_transform(svr_reg.predict(xsc.transform([[6.5]])).reshape(-1, 1))
print(pred)

# visualisation

plt.scatter(xsc.inverse_transform(x), ysc.inverse_transform(y), c='red', label='true values')
plt.plot(xsc.inverse_transform(x), ysc.inverse_transform(svr_reg.predict(x).reshape(-1, 1)), c='blue', label='predicted function')
plt.title("SVR Predictions")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
