import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("data/Salary_Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_train, y_train, c="red")
plt.plot(x_train, regressor.predict(x_train), c="blue")
plt.title("Salary vs Experience (training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(x_test, y_test, c="red")
plt.plot(x_train, regressor.predict(x_train), c="blue")
plt.title("Salary vs Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

