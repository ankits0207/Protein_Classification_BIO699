import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy


listOfListOfESS = [['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'DN', 'EF', 'FI', 'FR', 'GM', 'HG', 'HP', 'II', 'LRE', 'LVL',
                    'MD', 'MF', 'MN',
                    'NE', 'NR', 'NV', 'NY', 'RM', 'YI']]
listOfParams = ['0.45_1.7']

# Creating dataframes and dataseries
mesophilicDf = pd.read_csv('MesoSeq.csv')
mesoSeqDs = mesophilicDf['Sequence']
thermophilicDf = pd.read_csv('ThermoSeq.csv')
thermoSeqDs = thermophilicDf['Sequence']

numRowsMeso = mesophilicDf.shape[0]
numRowsThermo = thermophilicDf.shape[0]

print('Generating labels...')
# Generating labels
labels = []
for i in range(numRowsMeso):
    labels.append('0')
for i in range(numRowsThermo):
    labels.append('1')

print('Generating features...')
# Generating features
listOfListOfFeatureVectors = []
for listOfEss in listOfListOfESS:
    listOfFeatureVectors = []
    for ESS in listOfEss:
        tempList = []
        for i in range(numRowsMeso):
            mySequence = mesoSeqDs[i]
            tempList.append(mySequence.count(ESS))
        for i in range(numRowsThermo):
            mySequence = thermoSeqDs[i]
            tempList.append(mySequence.count(ESS))
        listOfFeatureVectors.append(tempList)
    listOfListOfFeatureVectors.append(listOfFeatureVectors)

listOfFeatureVectorsDf = []
idx0 = 0
for listOfEss in listOfListOfESS:
    featureVectorDf = pd.DataFrame()
    idx1 = 0
    for ESS in listOfEss:
        featureVectorDf[ESS] = listOfListOfFeatureVectors[idx0][idx1]
        idx1 += 1
    featureVectorDf['Label'] = labels
    listOfFeatureVectorsDf.append(featureVectorDf)
    idx0 += 1

modelIdx = 1
print('Running model...')
listOfAccuracy = []
for featureVectorDf in listOfFeatureVectorsDf:
    XTrain = featureVectorDf.loc[:, featureVectorDf.columns != 'Label'].values
    YTrain = featureVectorDf.loc[:, featureVectorDf.columns == 'Label'].values

    cvCount = 10
    clf = RandomForestClassifier(random_state=42, max_depth=52, max_features='sqrt', min_samples_leaf=2,
                                 min_samples_split=2, n_estimators=85, bootstrap=True)
    scores = cross_val_score(clf, XTrain, YTrain.ravel(), cv=cvCount)
    meanScore = sum(scores) / cvCount
    print(meanScore)
print('Done')
