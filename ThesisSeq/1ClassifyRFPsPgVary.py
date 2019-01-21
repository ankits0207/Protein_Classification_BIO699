import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy
import matplotlib.pyplot as plt


class MyClass:
    def __init__(self, numOfEstimators, score):
        self.numOfEstimators = numOfEstimators
        self.score = score

# Declaring the substrings to be used along with parameter configuration
listOfListOfESS = [['MN', 'NY'],
                   ['MN', 'NY', 'YI'],
                   ['HP', 'MN', 'NY', 'YI'],
                   ['FI', 'FR', 'GM', 'HP', 'MN', 'NV', 'NY', 'YI'],
                   ['DN', 'EF', 'FI', 'FR', 'GM', 'HG', 'HP', 'II', 'MN', 'NE', 'NR', 'NV', 'NY', 'YI'],
                   ['AEE', 'DAL', 'DEI', 'DII', 'LRE', 'LVL', 'MF'],
                   ['AEE', 'DAL', 'DEI', 'DII', 'LRE', 'LVL', 'MF', 'MN', 'NY'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'LRE', 'LVL', 'MF', 'MN', 'NY', 'RM', 'YI'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'LRE', 'LVL', 'MF', 'MN', 'NY', 'RM', 'YI', 'HP'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'LRE', 'LVL', 'MF', 'MN', 'NY', 'RM', 'YI', 'HP', 'MD'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'LRE', 'LVL', 'MF', 'MN', 'NY', 'RM', 'YI', 'HP', 'MD', 'FI',
                    'FR', 'GM', 'NV'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'DN', 'EF', 'FI', 'FR', 'GM', 'HG', 'HP', 'II', 'LRE', 'LVL',
                    'MD', 'MF', 'MN',
                    'NE', 'NR', 'NV', 'NY', 'RM', 'YI'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'DN', 'EF', 'FI', 'FR', 'GM', 'HG', 'HP', 'II', 'IK', 'IY', 'KN',
                    'LM', 'LRE',
                    'LVL', 'MD', 'MF', 'MN', 'NE', 'NR', 'NV', 'NY', 'RM', 'RR', 'TN', 'YI', 'YV'],
                   ['AAA', 'AEE', 'DAL', 'DEI', 'DII', 'DN', 'DT', 'EE', 'EF', 'EM', 'EY', 'FI', 'FP', 'FR', 'GM', 'HG',
                    'HP', 'II',
                    'IK', 'IP', 'IQ', 'IY', 'KN', 'LM', 'LRE', 'LVL', 'MD', 'MF', 'MN', 'MS', 'NE', 'NP', 'NR', 'NV',
                    'NY', 'RE', 'RI',
                    'RK', 'RM', 'RR', 'RS', 'SP', 'ST', 'TN', 'VH', 'YI', 'YP', 'YV'],
                   ['AAA', 'AEE', 'DAL', 'DD', 'DEI', 'DE', 'DII', 'DN', 'DS', 'DT', 'DV', 'EE', 'EF', 'EG', 'EK', 'EM',
                    'EV', 'EY',
                    'FI', 'FL', 'FP', 'FR', 'GM', 'GY', 'HG', 'HP', 'ID', 'II', 'IK', 'IP', 'IQ', 'IS', 'IV', 'IY',
                    'KI', 'KN', 'KR',
                    'KV', 'LM', 'LRE', 'LVL', 'MD', 'MF', 'MG', 'MN', 'MS', 'NE', 'NP', 'NR', 'NV', 'NY', 'RD', 'RE',
                    'RI', 'RK', 'RL',
                    'RM', 'RR', 'RS', 'SF', 'SI', 'SP', 'ST', 'TN', 'VH', 'YI', 'YP', 'YV'],
                   ['ADE', 'AEE', 'DAI', 'DEI', 'DII', 'DLG', 'EFR', 'EGM', 'EIA', 'EKI', 'ERL', 'ETL', 'GAA', 'GEV',
                    'IAE', 'IIG',
                    'IIV', 'ILA', 'LIL', 'LIS', 'LRE', 'LVL', 'REI', 'RII', 'VDL', 'VDV', 'VKI'],
                   ['ADE', 'ADI', 'AEE', 'ALR', 'DAI', 'DEI', 'DII', 'DLG', 'EFR', 'EGM', 'EIA', 'EKI', 'ERL', 'ETL',
                    'GAA', 'GEV',
                    'IAE', 'IIG', 'IIV', 'ILA', 'KAG', 'LAD', 'LIL', 'LIS', 'LRE', 'LVL', 'REI', 'RII', 'VDL', 'VDV',
                    'VKI'],
                   ['ADE', 'ADI', 'AEE', 'ALR', 'CL', 'DAI', 'DAL', 'DEI', 'DII', 'DLG', 'EFR', 'EGM', 'EIA', 'EKI',
                    'ERL', 'ETL',
                    'GAA', 'GAD', 'GEV', 'IAE', 'IIG', 'IIV', 'ILA', 'KAG', 'LAD', 'LIL', 'LIS', 'LRE', 'LVL', 'MF',
                    'REI', 'RII',
                    'VDL', 'VDV', 'VKI'],
                   ['AAA', 'AAG', 'AC', 'ADE', 'ADI', 'AEE', 'AGA', 'ALR', 'CL', 'DAI', 'DAL', 'DEI', 'DII', 'DLG',
                    'EFR', 'EGM',
                    'EIA', 'EKI', 'ERL', 'ETL', 'GAA', 'GAD', 'GC', 'GEV', 'HP', 'IAE', 'IIG', 'IIV', 'ILA', 'KAG',
                    'LAD', 'LIL',
                    'LIS', 'LRE', 'LVL', 'MF', 'MN', 'NY', 'REI', 'RII', 'RM', 'VDL', 'VDV', 'VKI', 'YI'],
                   ['AAA', 'AAG', 'AC', 'ADE', 'ADI', 'AEE', 'AGA', 'ALR', 'CL', 'DAI', 'DAL', 'DEI', 'DII', 'DLG',
                    'EFR', 'EF',
                    'EGM', 'EIA', 'EKI', 'ERL', 'ETL', 'FI', 'FR', 'GAA', 'GAD', 'GC', 'GEV', 'GM', 'HP', 'IAE', 'IIG',
                    'IIV',
                    'II', 'ILA', 'KAG', 'LAD', 'LIL', 'LIS', 'LRE', 'LVL', 'MD', 'MF', 'MN', 'NV', 'NY', 'REI', 'RII',
                    'RM',
                    'VDL', 'VDV', 'VKI', 'YI'],
                   ['AAA', 'AAG', 'AC', 'ADE', 'ADI', 'AEE', 'AGA', 'ALR', 'CL', 'DAI', 'DAL', 'DEI', 'DII', 'DLG',
                    'DN', 'EFR', 'EF', 'EGM',
                    'EIA', 'EKI', 'EM', 'ERL', 'ETL', 'FI', 'FP', 'FR', 'GAA', 'GAD', 'GC', 'GEV', 'GM', 'HG', 'HP',
                    'IAE',
                    'IIG', 'IIV', 'II', 'IK', 'ILA', 'IY', 'KAG', 'KN', 'LAD', 'LIL', 'LIS', 'LM', 'LRE', 'LVL', 'MD',
                    'MF',
                    'MN', 'MR', 'MS', 'NE', 'NR', 'NV', 'NY', 'REI', 'RII', 'RK', 'RM', 'RR', 'RS', 'SP', 'TN', 'VDL',
                    'VDV', 'VH', 'VKI', 'YI', 'YP', 'YV']

                   ]
listOfParams = ['0.5_2.4', '0.5_2.1', '0.5_2.0', '0.5_1.8', '0.5_1.7', '0.45_2.5', '0.45_2.4', '0.45_2.1',
                '0.45_2.0', '0.45_1.9', '0.45_1.8', '0.45_1.7', '0.45_1.6', '0.45_1.4', '0.45_1.3',
                '0.4_3', '0.4_2.75', '0.4_2.5', '0.4_2', '0.4_1.75', '0.4_1.5']

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
    print(str(modelIdx) + '-' + str(len(listOfParams)))
    XTrain = featureVectorDf.loc[:, featureVectorDf.columns != 'Label'].values
    YTrain = featureVectorDf.loc[:, featureVectorDf.columns == 'Label'].values

    listOfObjects = []
    cvCount = 10
    numOfEstimatorsUpperLimit = 125
    ne = 75
    while ne < numOfEstimatorsUpperLimit:
        clf = RandomForestClassifier(n_estimators=ne, random_state=42)
        scores = cross_val_score(clf, XTrain, YTrain.ravel(), cv=cvCount)
        meanScore = sum(scores) / cvCount
        listOfObjects.append(MyClass(ne, meanScore))
        ne += 1
    maxObj = None
    maxSc = 0
    for elt in listOfObjects:
        eltScore = elt.score
        if eltScore > maxSc:
            maxObj = elt
            maxSc = eltScore
    listOfAccuracy.append(maxSc)
    modelIdx += 1

ind = []
for i in range(len(listOfListOfESS)):
    print('Best accuracy: ' + str(listOfAccuracy[i]) + ' for ESSs: ' + str(listOfListOfESS[i]) +
          ', identifier: ' + str(i))
    ind.append(i)

x = numpy.arange(len(listOfParams))
plt.plot(x, listOfAccuracy)
plt.xticks(x, listOfParams)
plt.xlabel('Ps_Pg')
plt.ylabel('Accuracy')
plt.title('Sequence classification')
plt.show()

for listOfEss in listOfListOfESS:
    print(len(listOfEss))
print('Done')
