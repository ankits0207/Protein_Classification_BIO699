import pandas as pd
import numpy as np

mesoFV = pd.read_csv('MesoFV.csv')
thermoFV = pd.read_csv('ThermoFV.csv')

FV = mesoFV.append(thermoFV,ignore_index=True)
FV.to_csv('NetFV.csv', index=False)

myNumpyArr = FV.loc[:, FV.columns != 'pdbId'].values
Y_train = myNumpyArr[:, -1]
X_train = np.delete(myNumpyArr, myNumpyArr.shape[1]-1, axis=1)
np.save('X_train', X_train)
np.save('Y_train', Y_train)
print('Done')
