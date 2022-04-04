import pandas as pd
import matplotlib.pyplot as plt
import random

data = pd.read_csv('data/Ads_CTR_Optimisation.csv')

N = 400
d = 10
ads_selected = []
numbers_of_rewards_0 = [0] * d
numbers_of_rewards_1 = [0] * d
total_reward = 0

for n in range(N):
    ad = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = data.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward

plt.hist(ads_selected)
plt.title('Ads selected')
plt.xlabel('Ads')
plt.ylabel('Number of selections')
plt.show()
