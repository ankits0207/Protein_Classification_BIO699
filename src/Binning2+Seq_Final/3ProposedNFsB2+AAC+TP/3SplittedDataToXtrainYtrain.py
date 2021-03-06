import pandas as pd
import numpy as np

iter = 4

for i in range(iter):
    Df = pd.read_csv('MergedBalancedSubset_' + str(i) + '.csv')
    Y_train = Df.loc[:, Df.columns == 'label'].values
    X_train = Df.loc[:, Df.columns != 'pdbId']
    X_train = X_train.loc[:, X_train.columns != 'label'].values
    np.save('X_train_' + str(i), X_train)
    np.save('Y_train_' + str(i), Y_train)

print('Done')
