import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, matthews_corrcoef
import numpy as np

def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)

cvCount = 10
clf = RandomForestClassifier(random_state=42, max_depth=95, max_features='sqrt', min_samples_leaf=1,
                             min_samples_split=5, n_estimators=80, bootstrap=False)

myDataframe = pd.read_csv('NetFV.csv')

propsToBeDropped = ['dce', 'ebc', 'cfcc', 'sgc', 'dc2']
listOfColumnsToBeDropped = []
for propToBeDropped in propsToBeDropped:
    for i in range(10):
        listOfColumnsToBeDropped.append(propToBeDropped+'_'+str(i))
listOfColumnsToBeDropped.append('pdbId')
listOfColumnsToBeDropped.append('label')

mesoDf = myDataframe[myDataframe['label'] == 0]
thermoDf = myDataframe[myDataframe['label'] == 1]
mesoLabels = mesoDf['label']
thermoLabels = thermoDf['label']

m = 508
t = 575
iterations = 4

listOfMesoDfs = []
listOfMesoLabelDfs = []
for i in range(iterations):
    mesoSubDf = mesoDf.sample(n=m, replace=False, random_state=42)
    listOfMesoDfs.append(mesoSubDf)
    listOfMesoLabelDfs.append(mesoLabels[mesoSubDf.index])
    mesoDf = mesoDf.drop(index=mesoSubDf.index)
    mesoLabels = mesoLabels.drop(index=mesoSubDf.index)


skf = StratifiedKFold(n_splits=cvCount, shuffle=True)
thresholdRange = np.linspace(start=0.49, stop=0.51, num=20)

tl = []
pl = []
rl = []
f1l = []
accl = []
aurocl = []
mccl = []

for threshold in thresholdRange:
    print(threshold)
    overallp = 0
    overallr = 0
    overallf1 = 0
    overallacc = 0
    overallauroc = 0
    overallmcc = 0
    idx = 0
    for mesoDf in listOfMesoDfs:
        mergedDf = pd.concat([mesoDf, thermoDf], ignore_index=True)
        mergedDf.drop(columns=listOfColumnsToBeDropped, inplace=True)
        X = mergedDf.values
        mergedLabels = pd.concat([listOfMesoLabelDfs[idx], thermoLabels], ignore_index=True)
        Y = mergedLabels.values
        pcv = 0
        rcv = 0
        f1scv = 0
        acccv = 0
        auroccv = 0
        mcccv = 0
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            clf.fit(X_train, Y_train)
            predictions_prob = clf.predict_proba(X_test)
            predictions = getPredictionsGivenThreshold(predictions_prob, threshold)
            p = precision_score(Y_test, predictions)
            r = recall_score(Y_test, predictions)
            f1s = f1_score(Y_test, predictions)
            acc = accuracy_score(Y_test, predictions)
            fpr, tpr, thresholds = roc_curve(Y_test, predictions, pos_label=1)
            auroc = auc(fpr, tpr)
            mcc = matthews_corrcoef(Y_test, predictions)
            pcv += p
            rcv += r
            f1scv += f1s
            acccv += acc
            auroccv += auroc
            mcccv += mcc
        overallp += ((pcv*1.0)/cvCount)
        overallr += ((rcv * 1.0) / cvCount)
        overallf1 += ((f1scv * 1.0) / cvCount)
        overallacc += ((acccv * 1.0) / cvCount)
        overallauroc += ((auroccv * 1.0) / cvCount)
        overallmcc += ((mcccv * 1.0) / cvCount)
        idx += 1
    tl.append(threshold)
    pl.append(overallp/iterations)
    rl.append(overallr/iterations)
    f1l.append(overallf1/iterations)
    accl.append(overallacc/iterations)
    aurocl.append(overallauroc/iterations)
    mccl.append(overallmcc/iterations)

tDf = pd.DataFrame()
tDf['Threshold'] = tl
tDf['Precision'] = pl
tDf['Recall'] = rl
tDf['F1S'] = f1l
tDf['Accuracy'] = accl
tDf['AUROC'] = aurocl
tDf['MCC'] = mccl
tDf.to_csv('Threshold.csv', index=False)
print('Done')
