import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("data/50_Startups.csv.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values