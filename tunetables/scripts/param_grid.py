import math

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval, rand
except:
    print("Hyperopt not installed")

param_grid, param_grid_hyperopt = {}, {}
param_grid_hyperopt['ridge'] = {
    'max_iter': hp.randint('max_iter', 50, 500)
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'alpha': hp.loguniform('alpha', -5, math.log(5.0))}  # 'normalize': [False],
param_grid_hyperopt['lightgbm'] = {
    'num_leaves': hp.randint('num_leaves', 5, 50)
    , 'max_depth': hp.randint('max_depth', 3, 20)
    , 'learning_rate': hp.loguniform('learning_rate', -3, math.log(1.0))
    , 'n_estimators': hp.randint('n_estimators', 50, 2000)
    #, 'feature_fraction': 0.8,
    #, 'subsample': 0.2
    , 'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    , 'subsample': hp.uniform('subsample', 0.2, 0.8)
    , 'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.8)
    , 'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100])
    , 'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100])
}  # 'normalize': [False],
param_grid_hyperopt['logistic'] = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'none'])
    , 'max_iter': hp.randint('max_iter', 50, 500)
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'C': hp.loguniform('C', -5, math.log(5.0))}  # 'normalize': [False],

param_grid_hyperopt['random_forest'] = {'n_estimators': hp.randint('n_estimators', 20, 200),
            'max_features': hp.choice('max_features', ['auto', 'sqrt']),
            'max_depth': hp.randint('max_depth', 1, 45),
            'min_samples_split': hp.choice('min_samples_split', [5, 10])}
param_grid_hyperopt['knn'] = {'n_neighbors': hp.randint('n_neighbors', 1,16)
                            }
param_grid_hyperopt['gp'] = {
    'params_y_scale': hp.loguniform('params_y_scale', math.log(0.05), math.log(5.0)),
    'params_length_scale': hp.loguniform('params_length_scale', math.log(0.1), math.log(1.0))
}
param_grid_hyperopt['tabnet'] = {
    'n_d': hp.randint('n_d', 8, 64),
    'n_steps': hp.randint('n_steps', 3, 10),
    'max_epochs': hp.randint('max_epochs', 50, 200),
    'gamma': hp.uniform('relax', 1.0, 2.0),
    'momentum': hp.uniform('momentum', 0.01, 0.4),
}

param_grid_hyperopt['svm'] = {'C': hp.choice('C', [0.1,1, 10, 100]), 'gamma': hp.choice('gamma', ['auto', 'scale']),'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])}

# CatBoost hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
param_grid_hyperopt['catboost'] = {
    'learning_rate': hp.loguniform('learning_rate', math.log(math.pow(math.e, -5)), math.log(1)),
    'random_strength': hp.randint('random_strength', 1, 20),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', math.log(1), math.log(10)),
    'bagging_temperature': hp.uniform('bagging_temperature', 0., 1),
    'leaf_estimation_iterations': hp.randint('leaf_estimation_iterations', 1, 20),
    'iterations': hp.randint('iterations', 100, 4000), # This is smaller than in paper, 4000 leads to ram overusage
}

# XGB Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
param_grid_hyperopt['xgb'] = {
    'learning_rate': hp.loguniform('learning_rate', -7, math.log(1)),
    'max_depth': hp.randint('max_depth', 1, 10),
    'subsample': hp.uniform('subsample', 0.2, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
    'alpha': hp.loguniform('alpha', -16, 2),
    'lambda': hp.loguniform('lambda', -16, 2),
    'gamma': hp.loguniform('gamma', -16, 2),
    'n_estimators': hp.randint('n_estimators', 100, 4000), # This is smaller than in paper
}

param_grid['ridge'] = {'alpha': [0, 0.1, .5, 1.0, 2.0], 'fit_intercept': [True, False]} # 'normalize': [False],

def get_param_grids():
    return param_grid, param_grid_hyperopt