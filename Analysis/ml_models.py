import pandas as pd
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

from Analysis.anova import load_experiment_data, visualize_prediction_errors
from Analysis.hyper_param_opt import tune_mlp


def mlp_regression(X_train,
                   y_train,
                   hidden_layer_sizes=(100,),
                   max_iter=200,
                   activation="relu",
                   solver="adam",
                   learning_rate="constant",
                   learning_rate_init=0.001):
    regr = MLPRegressor(random_state=1,
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        activation=activation,
                        solver=solver,
                        learning_rate=learning_rate,
                        learning_rate_init=learning_rate_init)
    regr.fit(X_train, y_train)
    return regr


def linear_regression(X_train, y_train):
    regr = linear_model.LinearRegression()
    regr.fit(X=X_train, y=y_train)
    return regr


def split_data(df):
    X = df[["cpu", "memory", "batch_size", "model_name", "quant_scheme"]]
    y = df["inference_time"]
    return X, y


def get_predictions(regr, X_test):
    y_pred = regr.predict(X_test)
    return y_pred


if __name__ == "__main__":
    df = load_experiment_data()
    X, y = split_data(df)
    ohe = OneHotEncoder()
    ohe_X = ohe.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(ohe_X, y, random_state=1)
    # best = tune_mlp(X_train, y_train, seed=42, verbose=1, n_jobs=8)
    # print(best)
    # regr = mlp_regression(X_train=X_train,
    #                       y_train=y_train,
    #                       # hidden_layer_sizes=best["hidden_layer_sizes"],
    #                       # max_iter=best["max_iter"],
    #                       # activation=best["activation"],
    #                       # solver=best["solver"],
    #                       # learning_rate=best["learning_rate"],
    #                       # learning_rate_init=best["learning_rate_init"]
    #                       )

    regr = linear_regression(X_train, y_train)
    y_pred = get_predictions(regr, X_test)
    print(r2_score(y_true=y_test, y_pred=y_pred))

    visualize_prediction_errors(y_pred, y_test, file_name_to_save="plots/mlp_prediction_error.png")
