import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

XTrain = np.load('X_train.npy')
YTrain = np.load('Y_train.npy')

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
#     'max_depth': [int(x) for x in np.linspace(start=40, stop=45, num=5)],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [1, 2, 3],
#     'min_samples_split': [4, 5, 6],
#     'n_estimators': [int(x) for x in np.linspace(start=55, stop=60, num=5)]
# }

# # Create a based model
# rf = RandomForestClassifier()

# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cvCount, n_jobs=-1, verbose=2)
# # Fit the grid search to the data
# grid_search.fit(XTrain, YTrain.ravel())
# print(grid_search.best_params_)
# print('Done')
