import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, matthews_corrcoef

cvCount = 10
iterations = 4
clf = RandomForestClassifier(random_state=42, max_depth=50, max_features='sqrt', min_samples_leaf=1,
                             min_samples_split=6, n_estimators=85, bootstrap=False)

myDataframe = pd.read_csv('NetFV.csv')
myDataframe.drop(columns=['pdbId'], inplace=True)
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

pl = []
rl = []
f1l = []
accl = []
aurocl = []
mccl = []
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
    pl.append((pcv * 1.0) / cvCount)
    rl.append((rcv * 1.0) / cvCount)
    f1l.append((f1scv * 1.0) / cvCount)
    accl.append((acccv * 1.0) / cvCount)
    aurocl.append((auroccv * 1.0) / cvCount)
    mccl.append((mcccv * 1.0) / cvCount)
print('Precision=' + str(sum(pl)/iterations))
print('Recall=' + str(sum(rl)/iterations))
print('F1=' + str(sum(f1l)/iterations))
print('Acc=' + str(sum(accl)/iterations))
print('Auroc=' + str(sum(aurocl)/iterations))
print('Mcc=' + str(sum(mccl)/iterations))
print('Done')
