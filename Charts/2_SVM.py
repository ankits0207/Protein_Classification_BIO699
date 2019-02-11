import matplotlib.pylab as plt

plt.rc('font', family='serif')

# SVM-F1
plt.subplot(4, 1, 2)
clf = 'SVM'
metric = 'F1 score'
xlabel = 'experimental scenario'
ylabel = 'f1 score'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.6982339918739, 0.7077421369]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.502275880589698, 0.502275880589698, 0.502275880589698, 0.479375474726789, 0.465748227350321,
              0.318213719986702, 0.502275880589698, 0.502275880589698, 0.463596350692207, 0.502275880589698]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((0.35, 1))

plt.subplot(4, 1, 1)
# SVM-Accuracy
clf = 'SVM'
metric = 'Accuracy score'
xlabel = 'experimental scenario'
ylabel = 'accuracy score'
title = 'Network feature evaluation(Binning 1): ' + clf
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.698313934379376, 0.7058402284]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.843936326023717, 0.843936326023717, 0.843936326023717, 0.838181797729047, 0.831658094910266,
             0.797113275423357, 0.843936326023717, 0.843936326023717, 0.8335870213259, 0.843936326023717]
valueList0 = [i*100 for i in valueList0]
valueList1 = [i*100 for i in valueList1]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((65, 100))
plt.title(title)


# SVM-AUC
plt.subplot(4, 1, 3)
clf = 'SVM'
metric = 'Area under ROC curve'
xlabel = 'experimental scenario'
ylabel = 'area under ROC curve'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.773096615778798, 0.7737413734]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.745480938479654, 0.747249729731056, 0.743172126476224, 0.736884286851983, 0.73593579493253,
              0.750855048196135, 0.746552035295625, 0.746194721082625, 0.739512425534705, 0.746778067256692]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((0.7, 1))

# SVM-MCC
plt.subplot(4, 1, 4)
clf = 'SVM'
metric = 'MCC'
xlabel = 'experimental scenario'
ylabel = 'mcc'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.403126600889287, 0.4183465785]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.480788065639096, 0.480788065639096, 0.480788065639096, 0.45610768566792, 0.429598601742365,
              0.276399742204519, 0.480788065639096, 0.480788065639096, 0.434838831512441, 0.480788065639096]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim((0.25, 1))
plt.show()