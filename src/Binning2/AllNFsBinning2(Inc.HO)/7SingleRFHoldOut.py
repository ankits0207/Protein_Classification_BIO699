import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, matthews_corrcoef

cvCount = 10
clf = RandomForestClassifier(random_state=42, max_depth=95, max_features='sqrt', min_samples_leaf=1,
                             min_samples_split=5, n_estimators=80, bootstrap=False)

myDataframe = pd.read_csv('NetFV.csv')

propsToBeDropped = ['deg', 'clco', 'clce', 'dce', 'ebc', 'cfcc', 'sgc', 'cpl', 'cs', 'dc2']
listOfListOfColumnsToBeDropped = []
for propToBeDropped in propsToBeDropped:
    listOfColumnsToBeDropped = []
    for i in range(10):
        listOfColumnsToBeDropped.append(propToBeDropped+'_'+str(i))
    listOfListOfColumnsToBeDropped.append(listOfColumnsToBeDropped)

labelsDs = myDataframe['label']
myDataframe.drop(columns=['pdbId', 'label'], inplace=True)
skf = StratifiedKFold(n_splits=cvCount, random_state=42)
coll = []
pl = []
rl = []
f1l = []
accl = []
aurocl = []
mccl = []
idx = 1
for listOfColumnsToBeDropped in listOfListOfColumnsToBeDropped:
    print(idx)
    tempDf = myDataframe.copy()
    tempDf.drop(columns=listOfColumnsToBeDropped, inplace=True)
    X = tempDf.values
    Y = labelsDs.values
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
        predictions = clf.predict(X_test)
        predictions_prob = clf.predict_proba(X_test)
        p = precision_score(Y_test, predictions)
        r = recall_score(Y_test, predictions)
        f1s = f1_score(Y_test, predictions)
        acc = accuracy_score(Y_test, predictions)
        auroc = roc_auc_score(Y_test, predictions_prob[:, 1])
        mcc = matthews_corrcoef(Y_test, predictions)
        pcv += p
        rcv += r
        f1scv += f1s
        acccv += acc
        auroccv += auroc
        mcccv += mcc
    pl.append((pcv*1.0)/cvCount)
    rl.append((rcv * 1.0) / cvCount)
    f1l.append((f1scv * 1.0) / cvCount)
    accl.append((acccv * 1.0) / cvCount)
    aurocl.append((auroccv * 1.0) / cvCount)
    mccl.append((mcccv * 1.0) / cvCount)
    coll.append(''.join(listOfColumnsToBeDropped))
    idx += 1

finalDf = pd.DataFrame()
finalDf['HoldOutCol'] = coll
finalDf['Precision'] = pl
finalDf['Recall'] = rl
finalDf['F1'] = f1l
finalDf['Accuracy'] = accl
finalDf['AUROC'] = aurocl
finalDf['MCC'] = mccl
finalDf.to_csv('HoldOutComparisonNF.csv', index=False)
print('Done')
