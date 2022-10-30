import random

import numpy as np
from sklearn import linear_model, svm, ensemble
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import scipy
from scipy.stats import randint
from sklearn.neural_network import MLPRegressor


def tune_lr(X_train, y_train, seed=42, verbose=1, n_jobs=8):
    lr = linear_model.LogisticRegression()
    np.random.seed(seed)
    compatible = {
        "newton-cg": ["l2", "none"],
        "lbfgs": ["l2", "none"],
        "liblinear": ["l1", "l2"],
        "sag": ["l2", "none"],
        "saga": ["elasticnet", "l1", "l2", "none"]
    }
    best_params = None
    best_score = 0
    for k, v in compatible.items():
        param_distributions = {
            "solver": [k],
            "penalty": v,
            "C": scipy.stats.uniform(loc=0.0001, scale=10),  # exclude 0
            "l1_ratio": scipy.stats.uniform()  # must be optimized otherwise we can't finetune l1 penalty properly
        }
        search = HalvingRandomSearchCV(lr,
                                       param_distributions,
                                       random_state=seed,
                                       verbose=verbose,
                                       n_jobs=n_jobs).fit(X_train, y_train)
        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_params = search.best_params_
    return best_params


def tune_svm(X_train, y_train, seed=42, verbose=1, n_jobs=8):
    sv = svm.SVC(random_state=seed)
    np.random.seed(seed)
    param_distributions = {
        "C": scipy.stats.uniform(loc=0.0001, scale=10),  # exclude 0
        "kernel": ["linear", "poly", "rbf", "sigmoid"],  # removed 'precomputed' option because:
        # https://stackoverflow.com/questions/36306555/scikit-learn-grid-search-with-svm-regression
        "gamma": scipy.stats.uniform()
    }
    search = HalvingRandomSearchCV(sv,
                                   param_distributions,
                                   random_state=seed,
                                   verbose=verbose,
                                   n_jobs=n_jobs).fit(X_train, y_train)
    return search.best_params_


def tune_rf(X_train, y_train, seed=42, verbose=1, n_jobs=8):
    rf = ensemble.RandomForestClassifier(random_state=seed)
    np.random.seed(seed)
    param_distributions = {
                          "bootstrap": [True, False],
                          "max_depth": randint(1, len(y_train)),
                          "max_features": scipy.stats.uniform(),
                          "min_samples_leaf": scipy.stats.uniform(0, 0.5),
                          "min_samples_split": scipy.stats.uniform(),
                          "n_estimators": randint(10, 1000)
    }
    search = HalvingRandomSearchCV(rf,
                                   param_distributions,
                                   random_state=seed,
                                   n_jobs=n_jobs,
                                   verbose=verbose)
    search.fit(X_train, y_train)
    return search.best_params_


def tune_mlp(X_train, y_train, seed=42, verbose=1, n_jobs=8):
    mlp = MLPRegressor(random_state=seed)
    np.random.seed(seed)
    param_distributions = {
                          "hidden_layer_sizes": randint(10, 500),
                          "max_iter": randint(200, 500),
                          "activation": ["identity", "logistic", "tanh", "relu"],
                          "solver": ["lbfgs", "sgd", "adam"],
                          "learning_rate": ["constant", "invscaling", "adaptive"],
                          "learning_rate_init": scipy.stats.uniform(0.0001, 0.01)
    }
    search = HalvingRandomSearchCV(mlp,
                                   param_distributions,
                                   random_state=seed,
                                   verbose=verbose,
                                   n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search.best_params_
