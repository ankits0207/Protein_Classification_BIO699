import matplotlib.pylab as plt

plt.rc('font', family='serif')

# RF-F1
plt.subplot(4, 1, 2)
clf = 'Random Forest'
metric = 'F1 score'
xlabel = 'experimental scenario'
ylabel = 'f1 score'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.867035302, 0.8808298446]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.7040005943, 0.7190171822, 0.7255985439, 0.7249036004, 0.725653033,
             0.7255075026, 0.7361940994, 0.7376255712, 0.7421953418, 0.7483206542]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylim((0.65,1))
plt.ylabel(ylabel)

# RF-Accuracy
plt.subplot(4, 1, 1)
clf = 'Random Forest'
metric = 'Accuracy score'
xlabel = 'experimental scenario'
ylabel = 'accuracy score'
title = 'Network feature evaluation(Binning 2): ' + clf
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [86.68385262, 87.42183288]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [89.87585014, 90.22171039, 90.41372448, 90.2212683, 90.3752607,
             90.41343088, 90.68339397, 90.64478619, 90.8354774, 91.0665391]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim(80, 100)
plt.title(title)

# RF-AUC
plt.subplot(4, 1, 3)
clf = 'Random Forest'
metric = 'Area under ROC curve'
xlabel = 'experimental scenario'
ylabel = 'area under ROC curve'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.9440491204, 0.9443845237]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.8773004973, 0.878202871, 0.8769373708, 0.8826798463, 0.888565691,
             0.8883988864, 0.8831599385, 0.8864219744, 0.884308564, 0.8934654956]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim(0.8, 1)

# RF-MCC
plt.subplot(4, 1, 4)
clf = 'Random Forest'
metric = 'MCC'
xlabel = 'experimental scenario'
ylabel = 'mcc'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.7405014145, 0.7511039594]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.6886229463, 0.6985856005, 0.7066176728, 0.6992164839, 0.7042873276,
             0.7062872215, 0.713772498, 0.7126349284, 0.7197164537, 0.7271218229]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.ylim((0.6, 1))
plt.show()