import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd

iterations = 4
cvCount = 10
thresholdRange = np.linspace(start=0.46, stop=0.54, num=50)

best_params = [[False, 1, 185, 4, 'sqrt', 42], [False, 1, 264, 5, 'auto', 22],
               [False, 1, 57, 3, 'sqrt', 75], [False, 1, 45, 3, 'sqrt', 77]]

def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)


thresholdList = []
precisionList = []
recallList = []
aucList = []
accuracyList = []
mcList = []
for threshold in thresholdRange:
    print(threshold)
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
        overallPrecision = overallPrecision + (foldPrecision/cvCount)
        overallRecall = overallRecall + (foldRecall/cvCount)
        overallAuauc = overallAuauc + (foldAuauc/cvCount)
        overallAccuracy = overallAccuracy + (foldAccuracy/cvCount)
        overallMc = overallMc + (foldMc/cvCount)
    thresholdList.append(threshold)
    precisionList.append(overallPrecision/iterations)
    recallList.append(overallRecall/iterations)
    aucList.append(overallAuauc/iterations)
    accuracyList.append(overallAccuracy/iterations)
    mcList.append(overallMc/iterations)

df = pd.DataFrame()
df['Threshold'] = thresholdList
df['Precision'] = precisionList
df['Recall'] = recallList
df['AUROC'] = aucList
df['Accuracy'] = accuracyList
df['MC'] = mcList
df.to_csv('Thresholding.csv', index=False)
print('Done')
