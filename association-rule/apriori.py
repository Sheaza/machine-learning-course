import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

data = pd.read_csv('data/Market_Basket_Optimisation.csv', header=None)

# apriori model expects a certain format of the data (LIST of transactions)
transactions = []
for i in range(0, 7501):
    transactions.append([str(data.values[i, j]) for j in range(0, 20)])

print(transactions)

# min support = product at least 3 times a day so 21 times a week = 21/7501 divided by all transactions
# we can adjust the minimum confidence parameter but 0.2 is good enough
# min lift should be minimum 3 (from course, will read about it)

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

results = list(rules)
print(results)


def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))


resultsinDataFrame = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print(resultsinDataFrame.nlargest(n=10, columns='Lift'))

