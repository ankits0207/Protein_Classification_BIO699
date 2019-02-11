import matplotlib.pylab as plt

plt.rc('font', family='serif')

# # F1
# metric = 'F1 score'
# xlabel = 'classifiers'
# ylabel = 'f1 score'
# title = 'Impact of binning over ' + metric + ' for proposed NFs'
#
# orderingList = ['Naive Bayes', 'SVM', 'ANN', 'Random Forest']
# noBinningVal = [0.684286227, 0.6372947608, 0.6422255969, 0.7945464143]
# binning1Val = [0.509243236, 0.7077421369, 0.6828064202, 0.8748039066]
# binning2Val = [0.6323767141, 0.6629614573, 0.741520099663465, 0.8808298446]
#
# plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='No binning')
# plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='Binning 1')
# plt.plot(orderingList, binning2Val, color='k', marker='o', label='Binning 2')
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title(title)
# plt.legend(loc='upper left')
# plt.show()

# # Accuracy
# metric = 'Accuracy score'
# xlabel = 'classifiers'
# ylabel = 'accuracy score'
# title = 'Impact of binning over ' + metric + ' for proposed NFs'
#
# orderingList = ['Naive Bayes', 'SVM', 'ANN', 'Random Forest']
# noBinningVal = [0.5344627558, 0.6161148067, 0.5421899065, 0.8012981064]
# binning1Val = [0.5537649134, 0.7058402284, 0.6708416891, 0.8687578616]
# binning2Val = [0.6312813789, 0.6323013423, 0.743515720019943, 0.8742183288]
# noBinningVal = [i*100 for i in noBinningVal]
# binning1Val = [i*100 for i in binning1Val]
# binning2Val = [i*100 for i in binning2Val]
#
# plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='No binning')
# plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='Binning 1')
# plt.plot(orderingList, binning2Val, color='k', marker='o', label='Binning 2')
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title(title)
# plt.legend(loc='upper left')
# plt.show()

# # Area under ROC curve
# metric = 'Area under ROC curve'
# xlabel = 'classifiers'
# ylabel = 'Area under ROC curve'
# title = 'Impact of binning over ' + metric + ' for proposed NFs'
#
# orderingList = ['Naive Bayes', 'SVM', 'ANN', 'Random Forest']
# noBinningVal = [0.5295126449, 0.6182568266, 0.575333452, 0.8853423188]
# binning1Val = [0.6402865319, 0.7737413734, 0.7366326969, 0.9378271236]
# binning2Val = [0.6403795001, 0.6328851791, 0.814611538142178, 0.9443845237]
#
# plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='No binning')
# plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='Binning 1')
# plt.plot(orderingList, binning2Val, color='k', marker='o', label='Binning 2')
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.title(title)
# plt.legend(loc='upper left')
# plt.show()

# MCC
metric = 'MCC'
xlabel = 'classifiers'
ylabel = 'mcc'
title = 'Impact of binning over ' + metric + ' for proposed NFs'

orderingList = ['Naive Bayes', 'SVM', 'ANN', 'Random Forest']
noBinningVal = [0.026249688, 0.2309934692, 0, 0.6167129313]
binning1Val = [0.1475227945, 0.4183465785, 0.3501868518, 0.7413655455]
binning2Val = [0.26822857, 0.260729667, 0.495246597781015, 0.7511039594]

plt.plot(orderingList, noBinningVal, linestyle='--', color='r', marker='o', label='No binning')
plt.plot(orderingList, binning1Val, linestyle=':', color='b', marker='o', label='Binning 1')
plt.plot(orderingList, binning2Val, color='k', marker='o', label='Binning 2')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')
plt.show()