import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd

numRecordsMeso = 500
iterations = 4

numRecordsThermo = 575

mergedDf = pd.read_csv('MergedData.csv')
netFeatList = [['deg_0', 'deg_1', 'deg_2', 'deg_3', 'deg_4', 'deg_5', 'deg_6', 'deg_7', 'deg_8', 'deg_9'],
               ['clco_0', 'clco_1', 'clco_2', 'clco_3', 'clco_4', 'clco_5', 'clco_6', 'clco_7', 'clco_8', 'clco_9'],
               ['clce_0', 'clce_1', 'clce_2', 'clce_3', 'clce_4', 'clce_5', 'clce_6', 'clce_7', 'clce_8', 'clce_9'],
               ['dce_0', 'dce_1', 'dce_2', 'dce_3', 'dce_4', 'dce_5', 'dce_6', 'dce_7', 'dce_8', 'dce_9'],
               ['ebc_0', 'ebc_1', 'ebc_2', 'ebc_3', 'ebc_4', 'ebc_5', 'ebc_6', 'ebc_7', 'ebc_8', 'ebc_9'],
               ['cfcc_0', 'cfcc_1', 'cfcc_2', 'cfcc_3', 'cfcc_4', 'cfcc_5', 'cfcc_6', 'cfcc_7', 'cfcc_8', 'cfcc_9'],
               ['sgc_0', 'sgc_1', 'sgc_2', 'sgc_3', 'sgc_4', 'sgc_5', 'sgc_6', 'sgc_7', 'sgc_8', 'sgc_9']]

sfList = []
pList = []
rList = []
aucList = []
accList =[]
mcList = []
for netFeat in netFeatList:
    Df = mergedDf.copy()
    Df.drop(netFeat, axis=1, inplace=True)
    mesoDf = Df[Df['label'] == 0]
    thermoDf = Df[Df['label'] == 1]

    mesoDfsCV = []
    for i in range(iterations):
        mesoSubset = mesoDf.sample(n=numRecordsMeso, replace=False, random_state=42)
        mesoDfsCV.append(mesoSubset)
        mesoDf = mesoDf.drop(mesoSubset.index)

    thermoDfCV = thermoDf.sample(n=numRecordsThermo, replace=False, random_state=42)
    thermoDf = thermoDf.drop(thermoDfCV.index)

    for i in range(iterations):
        tempDf = pd.concat([mesoDfsCV[i], thermoDfCV])
        tempDf.to_csv('MergedBalancedSubset_' + str(i) + '.csv', index=False)

    for i in range(iterations):
        Df = pd.read_csv('MergedBalancedSubset_' + str(i) + '.csv')
        Y_train = Df.loc[:, Df.columns == 'label'].values
        X_train = Df.loc[:, Df.columns != 'pdbId']
        X_train = X_train.loc[:, X_train.columns != 'label'].values
        np.save('X_train_' + str(i), X_train)
        np.save('Y_train_' + str(i), Y_train)

    cvCount = 10
    thresholdRange = [0.495918367346939]

    best_params = [[False, 1, 185, 4, 'sqrt', 42], [False, 1, 264, 5, 'auto', 22],
                   [False, 1, 57, 3, 'sqrt', 75], [False, 1, 45, 3, 'sqrt', 77]]


    def getPredictionsGivenThreshold(myMatrix, th):
        myList = []
        for i in range(myMatrix.shape[0]):
            p1 = myMatrix[i, 1]
            if p1 >= th:
                myList.append(1)
            else:
                myList.append(0)
        return np.asarray(myList)

    for threshold in thresholdRange:
        overallPrecision = 0
        overallRecall = 0
        overallAuauc = 0
        overallAccuracy = 0
        overallMc = 0
        for i in range(iterations):
            X_train = np.load('X_train_' + str(i) + '.npy')
            Y_train = np.load('Y_train_' + str(i) + '.npy')
            skf = StratifiedKFold(n_splits=cvCount, random_state=42)
            foldPrecision = 0
            foldRecall = 0
            foldAuauc = 0
            foldAccuracy = 0
            foldMc = 0
            for train_index, test_index in skf.split(X_train, Y_train):
                X_tr, X_te = X_train[train_index], X_train[test_index]
                Y_tr, Y_te = Y_train[train_index], Y_train[test_index]
                bp = best_params[i]
                clf = RandomForestClassifier(bootstrap=bp[0], min_samples_leaf=bp[1], n_estimators=bp[2],
                                             min_samples_split=bp[3], max_features=bp[4], max_depth=bp[5],
                                             random_state=42).fit(X_tr, Y_tr.ravel())
                predictionsProb = clf.predict_proba(X_te)
                predictions = getPredictionsGivenThreshold(predictionsProb, threshold)
                precision = precision_score(Y_te, predictions)
                recall = recall_score(Y_te, predictions)
                fpr, tpr, thresholds = roc_curve(Y_te, predictions, pos_label=1)
                auroc = auc(fpr, tpr)
                accuracy = accuracy_score(Y_te, predictions)
                matthewsCoeff = matthews_corrcoef(Y_te, predictions)

                foldPrecision += precision
                foldRecall += recall
                foldAuauc += auroc
                foldAccuracy += accuracy
                foldMc += matthewsCoeff
            overallPrecision = overallPrecision + (foldPrecision / cvCount)
            overallRecall = overallRecall + (foldRecall / cvCount)
            overallAuauc = overallAuauc + (foldAuauc / cvCount)
            overallAccuracy = overallAccuracy + (foldAccuracy / cvCount)
            overallMc = overallMc + (foldMc / cvCount)
        sfList.append(''.join(netFeat))
        pList.append(overallPrecision/iterations)
        rList.append(overallRecall/iterations)
        aucList.append(overallAuauc/iterations)
        accList.append(overallAccuracy/iterations)
        mcList.append(overallMc/iterations)
        print('Dropping netFeat: ' + ''.join(netFeat) + ' p:' + str(overallPrecision / iterations) +
              ' r:' + str(overallRecall / iterations) + ' auc:' + str(overallAuauc / iterations) + ' acc:'
              + str(overallAccuracy / iterations) + ' Mc:' + str(overallMc/iterations))
DropDf = pd.DataFrame()
DropDf['NetFeat'] = sfList
DropDf['Precision'] = pList
DropDf['Recall'] = rList
DropDf['AUC'] = aucList
DropDf['Accuracy'] = accList
DropDf['MCC'] = mcList
DropDf.to_csv('DropNet.csv', index=False)
print('Done')
