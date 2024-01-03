import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# define probabilities
probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# create probability distribution
dists = [[p, 1.0 - p] for p in probs]
# calculate entropy for each distribution
ents = [entropy(d) / np.log(2) for d in dists]

plt.plot(probs, ents, marker='.')
plt.xticks(probs, [str(d) for d in dists])
plt.xlabel('Probability Distribution')
plt.ylabel('Normalized Entropy')
plt.show()
