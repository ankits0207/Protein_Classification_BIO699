from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

XTrain = np.load('X_train.npy')
YTrain = np.load('Y_train.npy')

cvCount = 10
clf = RandomForestClassifier(random_state=42, max_depth=38, max_features='sqrt', min_samples_leaf=1,
                             min_samples_split=5, n_estimators=74, bootstrap=False)
scores = cross_val_score(clf, XTrain, YTrain.ravel(), cv=cvCount)
meanScore = sum(scores) / cvCount
print(meanScore)
