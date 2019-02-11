import matplotlib.pylab as plt

plt.rc('font', family='serif')

# NB-F1
plt.subplot(4, 1, 2)
clf = 'Naive Bayes'
metric = 'F1 score'
xlabel = 'experimental scenario'
ylabel = 'f1 score'
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.682104938533092, 0.6323767141]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.364442728554555, 0.363534478542795, 0.365498967965239, 0.364428881644653, 0.364367959925181,
             0.364319163542867, 0.364334774161282, 0.364249511598264, 0.366226905182652, 0.365162034672804]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((0.3, 1))

# NB-Accuracy
plt.subplot(4, 1, 1)
clf = 'Naive Bayes'
metric = 'Accuracy score'
xlabel = 'experimental scenario'
ylabel = 'accuracy score'
title = 'Network feature evaluation(Binning 2): ' + clf
orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.577784843490494, 0.6312813789]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.245492201051559, 0.23858821372246, 0.246255559830185, 0.24280577659888, 0.2439566755084,
             0.242421149965241, 0.245107585666943, 0.243572060123784, 0.245879763680501, 0.247798419737088]

valueList0 = [i*100 for i in valueList0]
valueList1 = [i*100 for i in valueList1]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((20, 100))
plt.title(title)

# Naive Bayes-AUC
plt.subplot(4, 1, 3)
clf = 'Naive Bayes'
metric = 'Area under ROC curve'
xlabel = 'experimental scenario'
ylabel = 'area under ROC curve'
title = 'Network feature evaluation(Binning 2): ' + clf + ' (' + metric + ')'

orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.64799381991151, 0.6403795001]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.51041972455795, 0.506512232529284, 0.511655498296697, 0.508843713028111, 0.509192444109504,
             0.509023338667716, 0.510173419139231, 0.508950459838482, 0.513116269725864, 0.507802822612242]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim((0.45, 1))

# Naive Bayes-MCC
plt.subplot(4, 1, 4)
clf = 'Naive Bayes'
metric = 'MCC'
xlabel = 'experimental scenario'
ylabel = 'mcc'
title = 'Network feature evaluation(Binning 2): ' + clf + ' (' + metric + ')'

orderingList0 = ['All 10 NFs', 'Proposed NFs']
valueList0 = [0.14982198991526, 0.26822857]
orderingList1 = ['Hold out-CLCN', 'Hold out-EBC', 'Hold out-DC2',
                'Hold out-CS', 'Hold out-D', 'Hold out-SGC', 'Hold out-DC1', 'Hold out-CLCO', 'Hold out-CPL',
                'Hold out-CFC']
valueList1 = [0.04292857451237, 0.042322130852952, 0.050590274638109, 0.045365046134993, 0.044678737185741,
             0.043581016841315, 0.041254542895065, 0.043863511050073, 0.055525561196561, 0.047198638423002]

barlist0 = plt.bar(orderingList0[0], valueList0[0], hatch='/')
barlist1 = plt.bar(orderingList0[1], valueList0[1], hatch='x')
barlist2 = plt.bar(orderingList1, valueList1)
plt.ylabel(ylabel)
plt.ylim(0, 1)
plt.show()