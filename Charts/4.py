from matplotlib import pyplot as plt

plt.rc('font', family='serif')

# Proposed NFs + AAC + SP
xlabel = 'classifiers'
ylabel = 'evaluation metrics'
title = 'Comparison of proposed features - Proposed NFs(Binning 2) + AAC + SP '

orderingList = ['NB', 'SVM', 'ANN', 'RF']
accuracyList = [0.6100294219, 0.8237041364, 0.8834847698, 0.96394081]
f1List = [0.6977751752, 0.8304670991, 0.8890200743, 0.9657638839]
aucList = [0.7005757713, 0.8902333636, 0.9493962492, 0.991625608]
mccList = [0.2145047581, 0.6496457558, 0.7683050523, 0.9275052766]

plt.plot(orderingList, accuracyList, linestyle='--', color='r', marker='o', label='Accuracy')
plt.plot(orderingList, f1List, linestyle=':', color='b', marker='o', label='F1 score')
plt.plot(orderingList, aucList, linestyle = '-', color='g', marker='o', label='Area under ROC')
plt.plot(orderingList, mccList, color='k', marker='o', label='MCC')

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')
plt.show()