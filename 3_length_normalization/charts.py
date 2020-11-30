import matplotlib.pyplot as plt

plt.plot([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [22.4, 22.4, 22.4, 22.4, 22.4, 22.7, 22.7, 22.8, 22.8, 23.0, 23.0], )
plt.xlabel('α')
plt.ylabel('BLEU')
plt.savefig("fig_2.pdf")
plt.close()

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [21.1, 22.5, 23.0, 22.6, 23.1, 23.1, 22.9, 22.7, 22.9, 22.9])
plt.xlabel('k')
plt.ylabel('BLEU')
plt.savefig("fig_3.pdf")
