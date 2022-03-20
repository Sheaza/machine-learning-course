import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

np.set_printoptions(precision=2)

data = pd.read_csv("data/Data1.csv")
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# multiple linear regression

multi_reg = LinearRegression()
multi_reg.fit(x_train, y_train)

y_pred_multi = multi_reg.predict(x_test)
print(r2_score(y_test, y_pred_multi))

# polynomial regression

poly_feat = PolynomialFeatures(degree=4)
x_poly = poly_feat.fit_transform(x_train)
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y_train)

y_pred_poly = poly_reg.predict(poly_feat.transform(x_test))
print(r2_score(y_test, y_pred_poly))

# svm regression

y_s = y.reshape(len(y), 1)
x_train_s, x_test_s, y_train_s, y_test_s = train_test_split(x, y_s, test_size=0.2, random_state=0)


sc_X = StandardScaler()
sc_y = StandardScaler()
x_train_svr = sc_X.fit_transform(x_train_s)
y_train_svr = sc_y.fit_transform(y_train_s)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_train_svr, y_train_svr)

y_pred_svm = sc_y.inverse_transform(svr_reg.predict(sc_X.transform(x_test_s)).reshape(-1, 1))
print(r2_score(y_test_s, y_pred_svm))

# decision tree regression

dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(x_train, y_train)

y_pred_dt = dt_reg.predict(x_test)
print(r2_score(y_test, y_pred_dt))

# random forest regression

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(x_train, y_train)

y_pred_rf = rf_reg.predict(x_test)
print(r2_score(y_test, y_pred_rf))
