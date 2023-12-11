import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

alpha = 0.1

xs = [x for x in range(1, 21)]

mean = np.array([0.87459922, 0.89909355, 0.88544984, 0.88635505, 0.89650653,
       0.91743202, 0.91704746, 0.92907877, 0.94208681, 0.94195549,
       0.94597448, 0.94266848, 0.94771994, 0.95750365, 0.95928934,
       0.96073144, 0.96066553, 0.96350262, 0.98229442, 0.98260141])

std = np.array([0.0157866 , 0.02174476, 0.01790953, 0.01638137, 0.02600168,
       0.0185704 , 0.02203029, 0.02586326, 0.03674864, 0.0388361 ,
       0.03691317, 0.03972138, 0.03982836, 0.0315073 , 0.03575269,
       0.03689858, 0.03724805, 0.03813483, 0.01332493, 0.0118985 ])

ax_lp = sns.lineplot(x=xs, y=mean)

lower, upper = mean - std, mean + std
ax_lp.fill_between(xs, lower, upper, alpha=alpha)
ax_lp.set_xlabel("Query No.")
ax_lp.set_ylabel("Entropy")

plt.savefig("trec-bert-cb.pdf")
plt.show()
