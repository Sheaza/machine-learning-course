import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

data.drop('Name', axis=1, inplace=True)
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Embarked', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)
test_data.drop('PassengerId', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Embarked', axis=1, inplace=True)


le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
test_data['Sex'] = le.fit_transform(test_data['Sex'])
print(data.head(10))

data['SibSp&Parch'] = data['SibSp'] + data['Parch']
test_data['SibSp&Parch'] = test_data['SibSp'] + test_data['Parch']

data.drop('SibSp', axis=1, inplace=True)
data.drop('Parch', axis=1, inplace=True)
test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)


sc = StandardScaler()

data['Age'] = data['Age'].fillna(data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())

x_train = data.iloc[:, 1:].values
y_train = data.iloc[:, 0].values
x_test = test_data.values

x_train[:, 1:3] = sc.fit_transform(x_train[:, 1:3])
x_test[:, 2:4] = sc.fit_transform(x_test[:, 2:4])

classifier = LogisticRegression()

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(y_pred)