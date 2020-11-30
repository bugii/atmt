import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [21.1, 22.0, 22.4, 22.2, 22.2, 22.1, 22.0, 21.7, 21.6, 21.6], )
plt.xlabel('k')
plt.ylabel('BLEU')
plt.savefig("fig_1.pdf")
