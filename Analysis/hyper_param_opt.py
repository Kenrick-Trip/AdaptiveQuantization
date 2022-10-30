import numpy as np
from sklearn import linear_model
from sklearn.model_selection import RandomizedSearchCV
import scipy
from scipy.stats import randint
from sklearn.neural_network import MLPRegressor


def tune_elastic_net(X_train, y_train, seed=42, verbose=1, n_jobs=8):
    ridge = linear_model.ElasticNet(random_state=seed)
    np.random.seed(seed)
    param_distributions = {
        "alpha": scipy.stats.uniform(0.0001, 1000.0),
        "l1_ratio": scipy.stats.uniform(),
        "selection": ["random", "cyclic"],
        "max_iter": randint(100, 100000)
    }
    search = RandomizedSearchCV(ridge,
                                param_distributions,
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
    search = RandomizedSearchCV(mlp,
                                param_distributions,
                                random_state=seed,
                                verbose=verbose,
                                n_jobs=n_jobs)
    search.fit(X_train, y_train)
    return search.best_params_
