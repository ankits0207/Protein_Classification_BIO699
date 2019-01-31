import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

iter = 4
cvCount = 10

random_params = [[54, 4, 2, 'sqrt', 77, False], [69, 4, 2, 'sqrt', 23, False],
                 [213, 6, 2, 'sqrt', 47, False], [69, 4, 2, 'sqrt', 23, False]]

best_params = []
for i in range(iter):
    X_train = np.load('X_train_' + str(i) + '.npy')
    Y_train = np.load('Y_train_' + str(i) + '.npy')

    rp = random_params[i]

    if rp[2] == 1:
        # Create the parameter grid based on the results of random search
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(start=rp[0]-10, stop=rp[0]+10, num=20)],
            'min_samples_split': [rp[1]-1, rp[1], rp[1]+1],
            'min_samples_leaf': [rp[2], rp[2]+1],
            'max_features': [rp[3]],
            'max_depth': [int(x) for x in np.linspace(start=rp[4]-5, stop=rp[4]+5, num=10)],
            'bootstrap': [rp[5]]
        }
    else:
        # Create the parameter grid based on the results of random search
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(start=rp[0]-5, stop=rp[0]+5, num=5)],
            'min_samples_split': [rp[1]-1, rp[1], rp[1]+1],
            'min_samples_leaf': [rp[2] - 1, rp[2], rp[2]+1],
            'max_features': [rp[3]],
            'max_depth': [int(x) for x in np.linspace(start=rp[4]-5, stop=rp[4]+5, num=5)],
            'bootstrap': [rp[5]]
        }

    # Create a based model
    rf = RandomForestClassifier()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cvCount, n_jobs=-1, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train.ravel())
    best_params.append(grid_search.best_params_)
for best_param in best_params:
    print(best_param)
