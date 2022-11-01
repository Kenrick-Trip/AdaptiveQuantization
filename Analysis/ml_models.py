import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

from Analysis.anova import load_experiment_data, visualize_prediction_errors, split_data
from Analysis.hyper_param_opt import tune_mlp, tune_elastic_net


def mlp_regression(X_train, y_train, **kwargs):
    regr = MLPRegressor(random_state=1, **kwargs)
    regr.fit(X_train, y_train)
    return regr


def linear_regression(X_train, y_train):
    regr = linear_model.LinearRegression()
    regr.fit(X=X_train, y=y_train)
    return regr


def elastic_net(X_train, y_train, **kwargs):
    regr = linear_model.ElasticNet(**kwargs, random_state=1)
    regr.fit(X=X_train, y=y_train)
    return regr


def get_predictions(regr, X_test):
    y_pred = regr.predict(X_test)
    return y_pred


def run_with_and_without_tuning(regr_train_func, tune_func, data):
    X_train, X_test, y_train, y_test = data

    # get r2_score for defualt hyper parameters
    regr = regr_train_func(X_train=X_train, y_train=y_train)
    y_pred = get_predictions(regr, X_test)
    default_score = r2_score(y_true=y_test, y_pred=y_pred)
    tuned_score = None
    visualize_prediction_errors(y_pred, y_test, finetuned=False, model_name=regr_train_func.__name__)

    # find better hyper parameters, get r2_score for regressor trained with these params
    if tune_func is not None:
        best = tune_func(X_train, y_train, seed=42, verbose=1, n_jobs=8)
        regr = regr_train_func(X_train=X_train, y_train=y_train, **best)
        y_pred = get_predictions(regr, X_test)
        tuned_score = r2_score(y_true=y_test, y_pred=y_pred)
        visualize_prediction_errors(y_pred, y_test, finetuned=True, model_name=regr_train_func.__name__)

    return default_score, tuned_score


if __name__ == "__main__":
    df = load_experiment_data()
    X, y = split_data(df)
    ohe = OneHotEncoder()
    ohe_X = ohe.fit_transform(X)
    data = train_test_split(ohe_X, y, random_state=1)
    regr_train_funcs = [elastic_net, mlp_regression]
    tune_funcs = [tune_elastic_net, tune_mlp]
    for regr_train_func, tune_func in zip(regr_train_funcs, tune_funcs):
        default_score, tuned_score = run_with_and_without_tuning(regr_train_func, tune_func, data)
        print("{} default and tuned r_scores: {}, {}".format(regr_train_func.__name__, default_score, tuned_score))
