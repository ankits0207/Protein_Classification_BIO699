import matplotlib.pylab as plt
import numpy as np

plt.rcParams['figure.dpi'] = 125

plt.rc('font', family='serif', size=11.6, weight='bold')
width = 0.35
ind = np.arange(6)
ax1 = plt.subplot(1, 3, 1)
ax1.annotate('(a)', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
orderingList = ['AAC', 'SP', ' AAC+\nDP ', ' AAC+\nTP ', ' AAC+\nDP+\nTP ', ' AAC+\nSP ']
# F1
classifier = 'Random Forest'
metric = 'F1 score'
xlabel = 'various combinations of sequence features'
ylabel = 'f1 score'
netseqVal = [0.9420337134, 0.9495244536, 0.95006407, 0.9563122351, 0.956924104, 0.9657638839]
seqVal = [0.9368837871, 0.9218681499, 0.9470082711, 0.9414097294, 0.9390652009, 0.9485486407]
y_pos = np.arange(len(orderingList))
rects1 = ax1.barh(ind, seqVal, width, color='silver')
rects2 = ax1.barh(ind+width, netseqVal, width, color='black')
ax1.set_yticks(ind + width / 2)
ax1.set_yticklabels(orderingList, fontsize=8)
ax1.legend((rects1[0], rects2[0]), ('Sequence feature(s)', 'Sequence feature(s) + NF'), prop={'size': 8}, loc='upper right')
ax1.set_xlim([0.9,1])

# AUC
ax2 = plt.subplot(1, 3, 2)
ax2.annotate('(b)', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
netseqVal = [0.9849081216, 0.9911271174, 0.9926905626, 0.9916780853, 0.986367657, 0.991625608]
seqVal = [0.9787189193, 0.9864887473, 0.9844174156, 0.9848109619, 0.9797880138, 0.9849757237]
rects1 = ax2.barh(ind, seqVal, width, color='silver')
rects2 = ax2.barh(ind+width, netseqVal, width, color='black')
ax2.set_yticks(ind + width / 2)
ax2.set_yticklabels(orderingList, fontsize=8)
ax2.legend((rects1[0], rects2[0]), ('Sequence feature(s)', 'Sequence feature(s) + NF'), prop={'size': 8}, loc='upper right')
ax2.set_xlim([0.95,1])

# MCC
ax3 = plt.subplot(1, 3, 3)
ax3.annotate('(c)', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
netseqVal = [0.8786753033, 0.9013037644, 0.9146080789, 0.914951676, 0.8937883996, 0.9275052766]
seqVal = [0.8609291636, 0.8831131305, 0.8712361777, 0.8644961553, 0.8272037535, 0.8871363562]
rects1 = ax3.barh(ind, seqVal, width, color='silver')
rects2 = ax3.barh(ind+width, netseqVal, width, color='black')
ax3.set_yticks(ind + width / 2)
ax3.set_yticklabels(orderingList, fontsize=8)
ax3.legend((rects1[0], rects2[0]), ('Sequence feature(s)', 'Sequence feature(s) + NF'), prop={'size': 8}, loc='upper right')
ax3.set_xlim([0.8,1])
plt.show()
