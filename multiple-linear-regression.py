import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# we use the same simple linear regression class for multiple linear regression

dataset = pd.read_csv('data/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# we don't have to worry about the dummy variables lr class will take care of it

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# we don't have to specify that we use multiple features,
# and also we don't have to manually choose ones with the highest P values

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# single prediction

print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# equation

print(regressor.coef_)
print(regressor.intercept_)

