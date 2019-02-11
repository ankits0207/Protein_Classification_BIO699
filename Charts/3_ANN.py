import matplotlib.pylab as plt

plt.rc('font', family='serif')

plt.subplot(2, 2, 2)

# F1
classifier = 'ANN'
metric = 'F1 score'
xlabel = 'various combinations of sequence features'
ylabel = 'f1 score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.870941984, 0.8026967433, 0.8900820644, 0.8938781373, 0.8931897233, 0.8890200743]
seqVal = [0.846179496, 0.7756605994, 0.8859128486, 0.8809942839, 0.7884723482, 0.8659378135]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')

plt.subplot(2, 2, 1)

# Accuracy
classifier = 'ANN'
metric = 'Accuracy score'
xlabel = 'various combinations of sequence features'
ylabel = 'accuracy score'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.8632658359, 0.809326324, 0.8830520942, 0.8872187608, 0.8872165974, 0.8834847698]
netseqVal = [i*100 for i in netseqVal]
seqVal = [0.830324746, 0.7357455002, 0.8803383527, 0.8453453534, 0.8534534534, 0.8659378135]
seqVal = [i*100 for i in seqVal]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')

plt.subplot(2, 2, 3)

# AUC
classifier = 'ANN'
metric = 'Area under ROC curve'
xlabel = 'various combinations of sequence features'
ylabel = 'area under ROC curve'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.9306864791, 0.9140403811, 0.9470049909, 0.9477862976, 0.9505998185, 0.9493962492]
seqVal = [0.8836669165, 0.8616578947, 0.9393044943, 0.9131231354, 0.9202492048, 0.9269461585]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')

plt.subplot(2, 2, 4)

# MCC
classifier = 'ANN'
metric = 'MCC'
xlabel = 'various combinations of sequence features'
ylabel = 'mcc'
title = 'Evaluation of sequence features and overall classification: ' + classifier + ' (' + metric + ') '

orderingList = ['AAC', 'SP', 'AAC+DP', 'AAC+TP', 'AAC+DP+TP', 'AAC+SP']
netseqVal = [0.7271067147, 0.6371225549, 0.7670444866, 0.7748940216, 0.775764152, 0.7683050523]
seqVal = [0.65863716, 0.4804200798, 0.7645815434, 0.7331021638, 0.7576673453, 0.7293354643]
plt.plot(orderingList, netseqVal, color='r', marker='o', label='Proposed NFs(Binning 2) + Sequence Features')
plt.plot(orderingList, seqVal, linestyle=':', color='k', marker='o', label='Sequence Features')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend(loc='lower right')
plt.show()