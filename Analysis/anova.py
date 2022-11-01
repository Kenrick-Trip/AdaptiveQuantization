import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols


pd.set_option('display.max_columns', 500)


def regression_without_interaction(data):
    model = ols(
        "inference_time ~ C(cpu) + C(memory) + C(batch_size) + C(model_name) + C(quant_scheme) + C(processor)",
        data=data).fit()

    table = sm.stats.anova_lm(model)
    print(table)
    print(model.summary())
    return model


def regression_with_pairwise_interactions(data):
    model = ols(
        "inference_time ~ \
        C(cpu) + \
        C(memory) + \
        C(batch_size) + \
        C(model_name) + \
        C(quant_scheme) + \
        C(processor) +\
        C(cpu) * C(memory) + \
        C(cpu) * C(batch_size) + \
        C(cpu) *  C(model_name) + \
        C(cpu) *  C(quant_scheme) + \
        C(cpu) *  C(processor) + \
        C(memory) * C(batch_size) + \
        C(memory) * C(model_name) + \
        C(memory) * C(quant_scheme) + \
        C(memory) * C(processor) + \
        C(batch_size) * C(model_name) + \
        C(batch_size) * C(quant_scheme) + \
        C(batch_size) * C(processor) + \
        C(model_name) * C(quant_scheme) + \
        C(model_name) * C(processor) + \
        C(quant_scheme) * C(processor)",
        data=data).fit()

    table = sm.stats.anova_lm(model)
    print(table)
    print(model.summary())
    return model


def predict_inference_time(regression_model, X):
    # X: (cpu, memory, batch_size, model_name, quant_scheme)
    y_pred = regression_model.predict(X)
    return y_pred


def get_true_and_pred(dataframe, model):
    dataframe["predicted_inf_time"] = \
        predict_inference_time(model,
                               dataframe[["cpu", "memory", "batch_size", "model_name", "quant_scheme", "processor"]])
    dataframe.sort_values(axis=0, by=["inference_time"])
    y_pred = dataframe["predicted_inf_time"]
    y_true = dataframe["inference_time"]
    return y_pred, y_true


def visualize_prediction_errors(y_pred, y_true, model_name, finetuned):
    diff = y_true - y_pred
    score = float(r2_score(y_true=y_true, y_pred=y_pred))
    print("inference time error distribution mean: {:5.3f} and variance: {:5.3f}".format(np.mean(diff), np.var(diff)))
    diff.hist(bins=40)
    main_title_string = "Histogram of prediction errors for"
    if finetuned is None:
        plt.title('{} \n {}, R^2 score: {:5.3f}'.format(main_title_string, model_name, score))
    elif finetuned:
        plt.title('{} {} \n with hyper-parameter optimization, R^2 score: {:5.3f}'.format(main_title_string, model_name, score))
    elif not finetuned:
        plt.title('{} {} \n without hyper-parameter optimization, R^2 score: {:5.3f}'.format(main_title_string, model_name, score))
    plt.xlabel('Inference time (ms) prediction error')
    plt.ylabel('Frequency')
    plt.savefig("plots/{} prediction error finetune {}.png".format(model_name, finetuned))
    plt.show()


def load_experiment_data():
    hws = [1, 2, 3]
    reps = [1, 2, 3]
    combined_df = None
    for hw in hws:
        for rep in reps:
            df = pd.read_csv("data/hw{}_r{}_experiment_data.csv".format(hw, rep), header=None)
            df.columns = ["cpu", "memory", "batch_size", "model_name", "quant_scheme", "accuracy", "inference_time",
                          "model_size_mb"]
            df["processor"] = hw
            df["repetition"] = rep
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat((combined_df, df))
    return combined_df


def split_data(df, factors_to_use=["cpu", "memory", "batch_size", "model_name", "quant_scheme", "processor"]):
    X = df[factors_to_use]
    y = df["inference_time"]
    return X, y


if __name__ == "__main__":
    df = load_experiment_data()
    X, y = split_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    regr_train_data = X_train
    regr_train_data["inference_time"] = y_train
    regr_test_data = X_test
    regr_test_data["inference_time"] = y_test

    # Model 1
    model = regression_without_interaction(df)
    y_pred, y_true = get_true_and_pred(regr_test_data, model)
    visualize_prediction_errors(y_pred, y_true, finetuned=None, model_name="regression without interaction")

    # Model 2
    model = regression_with_pairwise_interactions(regr_train_data)
    y_pred, y_true = get_true_and_pred(regr_test_data, model)
    visualize_prediction_errors(y_pred, y_true, finetuned=None, model_name="regression with pairwise interaction")

