import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('data/Data.csv')

x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# logistic reg
log_clas = LogisticRegression(random_state=0)
log_clas.fit(x_train, y_train)

y_pred1 = log_clas.predict(x_test)

ac = accuracy_score(y_test, y_pred1)
print('linear reg: ', ac*100)

# knn
knn_clas = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_clas.fit(x_train, y_train)

y_pred2 = knn_clas.predict(x_test)

ac = accuracy_score(y_test, y_pred2)
print('knn: ', ac*100)

# svm linear
svc = SVC(kernel='linear', random_state=0)
svc.fit(x_train, y_train)

y_pred3 = svc.predict(x_test)

acc_score = accuracy_score(y_test, y_pred3)
print('svm linear: ', acc_score*100)

# svm rbf
svc = SVC(kernel='rbf', random_state=0)
svc.fit(x_train, y_train)

y_pred4 = svc.predict(x_test)

acc_score = accuracy_score(y_test, y_pred4)
print('svm rbf: ', acc_score*100)

# naive bayes
nb_clas = GaussianNB()
nb_clas.fit(x_train, y_train)

y_pred5 = nb_clas.predict(x_test)

acc_score = accuracy_score(y_test, y_pred5)
print('naive bayes: ', acc_score*100)

# decision tree
dt_clas = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_clas.fit(x_train, y_train)

y_pred6 = dt_clas.predict(x_test)

acc_score = accuracy_score(y_test, y_pred6)
print('decision tree: ', acc_score*100)

# random forest
rf_clas = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_clas.fit(x_train, y_train)

y_pred7 = rf_clas.predict(x_test)

acc_score = accuracy_score(y_test, y_pred7)
print('random forest: ', acc_score*100)
