import pandas as pd
import numpy as np

featuresToBeDropped = ['cfcc', 'clco', 'cpl']

mesoFV = pd.read_csv('MesoFV.csv')
thermoFV = pd.read_csv('ThermoFV.csv')

FV = mesoFV.append(thermoFV,ignore_index=True)

colsToBeDropped = []
for feature in featuresToBeDropped:
    for i in range(10):
        colsToBeDropped.append(feature + '_' + str(i))

FV.drop(columns=colsToBeDropped, inplace=True)
FV.to_csv('NetFV.csv', index=False)
myNumpyArr = FV.loc[:, FV.columns != 'pdbId'].values
Y_train = myNumpyArr[:, -1]
X_train = np.delete(myNumpyArr, myNumpyArr.shape[1]-1, axis=1)
np.save('X_train', X_train)
np.save('Y_train', Y_train)
print('Done')
