import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, matthews_corrcoef
import numpy as np

cvCount = 10
iterations = 4

clf = RandomForestClassifier(random_state=42, max_depth=98, max_features='sqrt', min_samples_leaf=1,
                             min_samples_split=5, n_estimators=80, bootstrap=False)

colsToBeDropped = ['pdbId']

def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)

myDataframe = pd.read_csv('NetFV.csv')
myDataframe.drop(columns=colsToBeDropped, inplace=True)
mesoDf = myDataframe[myDataframe['label'] == 0]
thermoDf = myDataframe[myDataframe['label'] == 1]

m = 508
t = 575
kf = StratifiedKFold(n_splits=cvCount, shuffle=True)

listOfMesoDfs = []
for i in range(iterations):
    mesoSubDf = mesoDf.sample(n=m, replace=False, random_state=42)
    listOfMesoDfs.append(mesoSubDf)
    mesoDf = mesoDf.drop(index=mesoSubDf.index)

thresholdRange = np.linspace(start=0.44, stop=0.56, num=50)

thresholdList = []
precisionList = []
recallList = []
f1list = []
aucList = []
accuracyList = []
mcList = []
for threshold in thresholdRange:
    print(threshold)
    pit = 0
    rit = 0
    f1it = 0
    accit = 0
    aurocit = 0
    mccit = 0
    for mesoSubDf in listOfMesoDfs:
        tempDf = pd.concat([mesoSubDf, thermoDf], axis=0, ignore_index=True)
        labels = tempDf['label']
        tempDf.drop(columns=['label'], inplace=True)
        tempDf = tempDf.values
        labels = labels.values

        pcv = 0
        rcv = 0
        f1scv = 0
        acccv = 0
        auroccv = 0
        mcccv = 0
        for train_index, test_index in kf.split(tempDf, labels):
            X_train, X_test = tempDf[train_index], tempDf[test_index]
            Y_train, Y_test = labels[train_index], labels[test_index]
            clf.fit(X_train, Y_train)
            predictions_prob = clf.predict_proba(X_test)
            predictions = getPredictionsGivenThreshold(predictions_prob, threshold)
            pcv += precision_score(Y_test, predictions)
            rcv += recall_score(Y_test, predictions)
            f1scv += f1_score(Y_test, predictions)
            acccv += accuracy_score(Y_test, predictions)
            fpr, tpr, thresholds = roc_curve(Y_test, predictions, pos_label=1)
            auroccv += auc(fpr, tpr)
            mcccv += matthews_corrcoef(Y_test, predictions)
        pit += (pcv/cvCount)
        rit += (rcv/cvCount)
        f1it += (f1scv/cvCount)
        accit += (acccv/cvCount)
        aurocit += (auroccv/cvCount)
        mccit += (mcccv/cvCount)
    thresholdList.append(threshold)
    precisionList.append(pit/iterations)
    recallList.append(rit/iterations)
    f1list.append(f1it/iterations)
    accuracyList.append(accit/iterations)
    aucList.append(aurocit/iterations)
    mcList.append(mccit/iterations)

df = pd.DataFrame()
df['Threshold'] = thresholdList
df['Precision'] = precisionList
df['F1'] = f1list
df['Recall'] = recallList
df['AUROC'] = aucList
df['Accuracy'] = accuracyList
df['MC'] = mcList
df.to_csv('Thresholding.csv', index=False)
print('Done')
