import matplotlib.pylab as plt

plt.rc('font', family='serif', size=14, weight='bold')
plt.rcParams['figure.dpi'] = 125

ax1 = plt.subplot(1, 3, 1)
ax1.annotate('(a)', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
# F1
metric = 'F1 score'
xlabel = 'classifiers'
ylabel = 'f1 score'

orderingList = ['NB', 'SVM', 'ANN', 'RF']
noBinningVal = [0.684286227, 0.6372947608, 0.6422255969, 0.7945464143]
binning1Val = [0.509243236, 0.7077421369, 0.6828064202, 0.8748039066]
binning2Val = [0.6323767141, 0.6629614573, 0.741520099663465, 0.8808298446]

plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='AVG')
plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='DSA')
plt.plot(orderingList, binning2Val, color='k', marker='o', label='DSF')
plt.legend(loc='upper left')

# Area under ROC curve
ax2 = plt.subplot(1, 3, 2)
ax2.annotate('(b)', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
orderingList = ['NB', 'SVM', 'ANN', 'RF']
noBinningVal = [0.5295126449, 0.6182568266, 0.575333452, 0.8853423188]
binning1Val = [0.6402865319, 0.7737413734, 0.7366326969, 0.9378271236]
binning2Val = [0.6403795001, 0.6328851791, 0.814611538142178, 0.9443845237]

plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='AVG')
plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='DSA')
plt.plot(orderingList, binning2Val, color='k', marker='o', label='DSF')
plt.legend(loc='upper left')

# MCC
ax3 = plt.subplot(1, 3, 3)
ax3.annotate('(c)', xy=(1, 0), xycoords='axes fraction', fontsize=10,
                horizontalalignment='right', verticalalignment='bottom')
orderingList = ['NB', 'SVM', 'ANN', 'RF']
noBinningVal = [0.026249688, 0.2309934692, 0, 0.6167129313]
binning1Val = [0.1475227945, 0.4183465785, 0.3501868518, 0.7413655455]
binning2Val = [0.26822857, 0.260729667, 0.495246597781015, 0.7511039594]
plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='AVG')
plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='DSA')
plt.plot(orderingList, binning2Val, color='k', marker='o', label='DSF')
plt.legend(loc='upper left')
plt.show()
