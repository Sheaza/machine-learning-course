import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/Social_Network_Ads.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc = StandardScaler()
x_test = sc.fit_transform(x_test)
x_train = sc.transform(x_train)

# minkowski and p=2 are for euclidean distance
knn_clas = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_clas.fit(x_train, y_train)

y_pred = knn_clas.predict(x_test)
print(knn_clas.predict(sc.transform([[30, 87000]])))
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(x_test), 1)), axis=1))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

acc_score = accuracy_score(y_test, y_pred)
print(acc_score)
