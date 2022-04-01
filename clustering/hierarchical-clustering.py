import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('data/Mall_Customers.csv')

# we are taking only 2 columns so we can visualise clusters, normally it would be [:, 1:] with encoding on categorical feature
x = data.iloc[:, [3, 4]].values

# ward is a method of minimizing variance inside clusters
dendogram = sch.dendrogram(sch.linkage(x, method='ward'))

plt.title('Dendogram')
plt.xlabel('Customers- observation points')
plt.ylabel('Euclidean distances')
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred = hc.fit_predict(x)
print(y_pred)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual income')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()


