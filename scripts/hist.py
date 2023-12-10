import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.1

xs = [x for x in range(1, 21)]

mean = np.array([0.86760778, 0.87858424, 0.88077679, 0.86163239, 0.81954582,
                 0.77164866, 0.73901381, 0.71657046, 0.69305419, 0.67968679,
                 0.66051145, 0.63197696, 0.64160798, 0.62668788, 0.63821238,
                 0.61917439, 0.6141192, 0.59730116, 0.61364753, 0.60612106])

std = np.array([0.03250474, 0.02741349, 0.02161812, 0.03388342, 0.05307041,
                0.0683287, 0.08058477, 0.0814525, 0.08876366, 0.09271208,
                0.08658591, 0.09662995, 0.07775582, 0.08675114, 0.08753616,
                0.07503812, 0.08467618, 0.08312343, 0.06574987, 0.06671644])

ax_lp = sns.lineplot(x=xs, y=mean)

lower, upper = mean - std, mean + std
ax_lp.fill_between(xs, lower, upper, alpha=alpha)
ax_lp.set_xlabel("Query No.")
ax_lp.set_ylabel("Entropy")

plt.show()
