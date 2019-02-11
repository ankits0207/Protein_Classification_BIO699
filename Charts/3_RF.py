import matplotlib.pylab as plt

plt.rc('font', family='serif')

plt.subplot(2, 2, 1)
# Accuracy
classifier = 'Random Forest'
metric = 'Accuracy score'
xlabel = 'various combinations of sequence features'
ylabel = 'accuracy score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.9388369678, 0.9488317757, 0.955352198, 0.955804344, 0.9469647802, 0.96394081]
netseqVal = [i*100 for i in netseqVal]
seqVal = [0.9302612482, 0.9416817852, 0.9357265965, 0.9323521408, 0.9142857143, 0.9442857143]
seqVal = [i*100 for i in seqVal]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')

plt.subplot(2, 2, 2)
# F1
classifier = 'Random Forest'
metric = 'F1 score'
xlabel = 'various combinations of sequence features'
ylabel = 'f1 score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.9420337134, 0.9495244536, 0.95006407, 0.9563122351, 0.956924104, 0.9657638839]
seqVal = [0.9368837871, 0.9218681499, 0.9470082711, 0.9414097294, 0.9390652009, 0.9485486407]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')


plt.subplot(2, 2, 3)
# AUC
classifier = 'Random Forest'
metric = 'Area under ROC curve'
xlabel = 'various combinations of sequence features'
ylabel = 'area under ROC curve'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.9849081216, 0.9911271174, 0.9926905626, 0.9916780853, 0.986367657, 0.991625608]
seqVal = [0.9787189193, 0.9864887473, 0.9844174156, 0.9848109619, 0.9797880138, 0.9849757237]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')

plt.subplot(2, 2, 4)
# MCC
classifier = 'Random Forest'
metric = 'MCC'
xlabel = 'various combinations of sequence features'
ylabel = 'mcc'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.8786753033, 0.9013037644, 0.9146080789, 0.914951676, 0.8937883996, 0.9275052766]
seqVal = [0.8609291636, 0.8831131305, 0.8712361777, 0.8644961553, 0.8272037535, 0.8871363562]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='upper left')
plt.show()