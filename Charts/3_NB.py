import matplotlib.pylab as plt

plt.rc('font', family='serif')

plt.subplot(2, 2, 2)

# F1
classifier = 'Naive Bayes'
metric = 'F1 score'
xlabel = 'various combinations of sequence features'
ylabel = 'f1 score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.694610712, 0.6879332309, 0.7168602905, 0.6998339246, 0.6991925254, 0.6977751752]
seqVal = [0.7076958037, 0.7517502154, 0.7086764665, 0.7090671223, 0.7087339918, 0.7792451659]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')

plt.subplot(2, 2, 1)

# Accuracy
classifier = 'Naive Bayes'
metric = 'Accuracy score'
xlabel = 'various combinations of sequence features'
ylabel = 'accuracy score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.5918721876, 0.5946824161, 0.6174260125, 0.5411496192, 0.5397542402, 0.6100294219]
netseqVal = [i*100 for i in netseqVal]
seqVal = [0.5476233672, 0.7679170993, 0.5497550798, 0.5526190476, 0.5519047619, 0.7930512288]
seqVal = [i*100 for i in seqVal]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')

plt.subplot(2, 2, 3)

# AUC
classifier = 'Naive Bayes'
metric = 'Area under ROC curve'
xlabel = 'various combinations of sequence features'
ylabel = 'area under ROC curve'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.6685169389, 0.6721067756, 0.7625452208, 0.7894163642, 0.7998219903, 0.7005757713]
seqVal = [0.5, 0.7757032668, 0.5023548316, 0.5054597195, 0.5046052502, 0.8007704174]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')

plt.subplot(2, 2, 4)

# MCC
classifier = 'Naive Bayes'
metric = 'MCC'
xlabel = 'various combinations of sequence features'
ylabel = 'mcc'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.1756175274, 0.1785321362, 0.2424669106, 0, 0, 0.2145047581]
seqVal = [0, 0.5610609149, 0.02419863541, 0.06320837545, 0.05131634288, 0.6108725499]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')
plt.show()