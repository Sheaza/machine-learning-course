import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

data = pd.read_csv('data/Mall_Customers.csv')

# we are taking only 2 columns so we can visualise clusters, normally it would be [:, 1:] with encoding on categorical feature
x = data.iloc[:, [3, 4]].values


