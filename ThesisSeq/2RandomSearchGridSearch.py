import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
    print(str(modelIdx) + '-' + str(len(listOfParams)))
    XTrain = featureVectorDf.loc[:, featureVectorDf.columns != 'Label'].values
    YTrain = featureVectorDf.loc[:, featureVectorDf.columns == 'Label'].values

    cvCount = 10

    # Specifying grid and performing random search
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=200, num=100)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 100, num=50)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestClassifier()
    print('Searching...')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=100, cv=cvCount, verbose=2,
                                   random_state=42)

    rf_random.fit(XTrain, YTrain.ravel())
    print(rf_random.best_params_)


    # # Create the parameter grid based on the results of random search -- Grid Search
    # param_grid = {
    #     'bootstrap': [False],
    #     'max_depth': [int(x) for x in np.linspace(start=50, stop=60, num=5)],
    #     'max_features': ['sqrt'],
    #     'min_samples_leaf': [1, 2, 3],
    #     'min_samples_split': [2, 3, 4],
    #     'n_estimators': [int(x) for x in np.linspace(start=80, stop=90, num=10)]
    # }
    #
    # # Create a based model
    # rf = RandomForestClassifier()
    #
    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cvCount, n_jobs=-1, verbose=2)
    # # Fit the grid search to the data
    # grid_search.fit(XTrain, YTrain.ravel())
    # print(grid_search.best_params_)
print('Done')
