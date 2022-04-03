import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

data = pd.read_csv('data/Ads_CTR_Optimisation.csv')

# it doesn't work with only 500 rounds
N = 1000
d = 10

ads_selected = []
numbers_of_selections = [0] * d
sum_of_reward = [0] * d
total_reward = 0
for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:
            average_reward = sum_of_reward[i] / numbers_of_selections[i]
            confidence_interval = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + confidence_interval
        else:
            upper_bound = 1e400  # we have to make sure that each ad is selected
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = data.values[n, ad]
    sum_of_reward[ad] += reward
    total_reward += reward

plt.hist(ads_selected)
plt.title('Ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of selections')
plt.show()