import matplotlib.pyplot as plt
from pylab import text

plt.rc('font', family='serif', size=14, weight='bold')


orderingList = ['Avg', 'Bin 1', 'Bin 2', 'Bin 3', 'Bin 4', 'Bin 5', 'Bin 6',
                'Bin 7', 'Bin 8', 'Bin 9', 'Bin 10']
m = [0.030, 0, 0.014098820396882, 0.097686240541678, 0.20556752869811, .226870695030315, 0.342664083424485,
      0.399966242518617, 0, 0, 0]
t = [0.027, 0, 0.00639596625172, 0.095071336163261, 0, 0, 0, 0, 0, 0, 0]

ax1 = plt.subplot(1, 2, 1)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.set_xticklabels(labels=orderingList,rotation=90)
ax1.annotate('A', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')

plt.plot(orderingList, m, linestyle='--', color='r', marker='o', label='3D3R: mesophilic protein')
plt.plot(orderingList, t, linestyle=':', color='b', marker='o', label='3VPB: thermophilic protein')
plt.legend(loc='upper left')

m = [0.0630, 0, 0.813880126182965, 0.018927444794953, 0.050473186119874, 0.028391167192429, 0.031545741324921,
      0.012618296529969, 0.04416403785489, 0, 0]
t = [0.0658, 0, 0.998178506375228, 0.001821493624772, 0, 0, 0, 0, 0, 0, 0]

ax2 = plt.subplot(1, 2, 2)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_xticklabels(labels=orderingList,rotation=90)
ax2.annotate('B', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
plt.plot(orderingList, m, linestyle='--', color='r', marker='o', label='2WH7: mesophilic protein')
plt.plot(orderingList, t, linestyle=':', color='b', marker='o', label='1MTP: thermophilic protein')
plt.legend(loc='upper right')
plt.show()
