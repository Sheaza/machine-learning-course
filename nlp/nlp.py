import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nltk.download('stopwords')


data = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

corpus = []

for i in range(1000):
    review = re.sub("[^a-zA-Z]", ' ', data['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data["Liked"].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

nb_clas = GaussianNB()
nb_clas.fit(x_train, y_train)

y_pred = nb_clas.predict(x_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(x_test), 1)), axis=1))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

acc_score = accuracy_score(y_test, y_pred)
print(acc_score)