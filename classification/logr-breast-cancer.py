import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
data = pd.read_csv('data/breast_cancer.csv')

x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

log_clas = LogisticRegression(random_state=0)
log_clas.fit(x_train, y_train)

y_pred = log_clas.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print(cm)
print(ac)

acs = cross_val_score(estimator=log_clas, X=x_train, y=y_train, cv=10)
print('Accuracy: ', acs.mean()*100)
print('Std: ', acs.std()*100)
