import matplotlib.pylab as plt

plt.rc('font', family='serif')

# ANN-F1
plt.subplot(4, 1, 2)
clf = 'ANN'
metric = 'F1 score'
xlabel = 'experimental scenario'
ylabel = 'f1 score'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.738349138741792, 0.741520099663465]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.542171186107506, 0.537588916737514, 0.515949414924012, 0.534325676868694, 0.491639381196657,
             0.535421418342731, 0.509330653939168, 0.54648797929414, 0.52210749515292, 0.538780777210151]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylim((0.45, 1))
plt.ylabel(ylabel)

# ANN-Accuracy
plt.subplot(4, 1, 1)
clf = 'ANN'
metric = 'Accuracy score'
xlabel = 'experimental scenario'
ylabel = 'accuracy score'
title = 'Network feature evaluation(Binning 2): ' + clf
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [73.5928474345906, 74.3515720019943]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [84.8485645120286, 84.7731116825615, 84.2369989403419, 84.7342091823534, 83.9298972064173,
             84.6969220420128, 83.8523858055314, 85.196187480736, 84.2725256084035, 84.8494464355217]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.title(title)
plt.ylabel(ylabel)
plt.ylim((70, 100))

# ANN-AUC
plt.subplot(4, 1, 3)
clf = 'ANN'
metric = 'Area under ROC curve'
xlabel = 'experimental scenario'
ylabel = 'area under ROC curve'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.813574564369002, 0.814611538142178]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.744551403075781, 0.74722880985425, 0.738312842503545, 0.746590281396995, 0.729057463463513,
             0.743123331690771, 0.735827084760934, 0.744710507646328, 0.74640088058504, 0.748710343390125]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((0.7, 1))

# ANN-MCC
plt.subplot(4, 1, 4)
clf = 'ANN'
metric = 'MCC'
xlabel = 'experimental scenario'
ylabel = 'mcc'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.478069181610565, 0.495246597781015]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.507682689419754, 0.504155120131176, 0.483554682941229, 0.501161410508739, 0.467975754792656,
             0.500949480987001, 0.470361915991511, 0.517307007650877, 0.487829002255377, 0.507344081923804]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim(0.4, 1)
plt.show()