import matplotlib.pylab as plt

plt.rc('font', family='serif')

plt.subplot(2, 2, 2)

# F1
classifier = 'SVM'
metric = 'F1 score'
xlabel = 'various combinations of sequence features'
ylabel = 'f1 score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.8199543136, 0.6842642195, 0.8601544924, 0.8727585172, 0.8770607536, 0.8304670991]
seqVal = [0.8494831621, 0.7576933102, 0.8738440331, 0.8873528796, 0.8887259303, 0.8497220367]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')

plt.subplot(2, 2, 1)
# Accuracy
classifier = 'SVM'
metric = 'Accuracy score'
xlabel = 'various combinations of sequence features'
ylabel = 'accuracy score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.8125519211, 0.684440983, 0.8539697992, 0.8672204915, 0.8649078401, 0.8237041364]
netseqVal = [i*100 for i in netseqVal]
seqVal = [0.8329689768, 0.7682048287, 0.864853955, 0.8802380952, 0.8821428571, 0.8416883004]
seqVal = [i*100 for i in seqVal]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')


plt.subplot(2, 2, 3)

# AUC
classifier = 'SVM'
metric = 'Area under ROC curve'
xlabel = 'various combinations of sequence features'
ylabel = 'area under ROC curve'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.8795981549, 0.760331216, 0.9181010284, 0.9287728373, 0.9340591349, 0.8902333636]
seqVal = [0.8298060184, 0.7745065033, 0.8654617934, 0.8815824359, 0.8841795866, 0.8418455838]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')

plt.subplot(2, 2, 4)

# MCC
classifier = 'SVM'
metric = 'MCC'
xlabel = 'various combinations of sequence features'
ylabel = 'mcc'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.6272176316, 0.3769386589, 0.7097719148, 0.7361611151, 0.7294416896, 0.6496457558]
seqVal = [0.6640992681, 0.5546844171, 0.7303878158, 0.7595160269, 0.7642518609, 0.6847934733]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')
plt.show()