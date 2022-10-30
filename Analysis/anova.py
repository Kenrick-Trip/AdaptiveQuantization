import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.max_columns', 500)


def regression_without_interaction(data):
    model = ols(
        "inference_time ~ C(cpu) + C(memory) + C(batch_size) + C(model_name) + C(quant_scheme)",
        data=data).fit()

    table = sm.stats.anova_lm(model)
    print(table)
    print(model.summary())
    return model


def regression_with_pairwise_interactions(data):
    model = ols(
        "inference_time ~ \
        C(cpu) * C(memory) + \
        C(cpu) * C(batch_size) + \
        C(cpu) *  C(model_name) + \
        C(cpu) *  C(quant_scheme) + \
        C(memory) * C(batch_size) + \
        C(memory) * C(model_name) + \
        C(memory) * C(quant_scheme) + \
        C(batch_size) * C(model_name) + \
        C(batch_size) * C(quant_scheme) + \
        C(model_name) * C(quant_scheme)",
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
        predict_inference_time(model, dataframe[["cpu", "memory", "batch_size", "model_name", "quant_scheme"]])
    dataframe.sort_values(axis=0, by=["inference_time"])
    y_pred = dataframe["predicted_inf_time"]
    y_true = dataframe["inference_time"]
    return y_pred, y_true


def visualize_prediction_errors(y_pred, y_true, file_name_to_save="regression_prediction_error.png"):
    diff = y_true - y_pred
    diff.hist(bins=40)
    plt.title('Histogram of prediction errors')
    plt.xlabel('Inference time (ms) prediction error')
    plt.ylabel('Frequency')
    plt.savefig(file_name_to_save)
    plt.show()


def load_experiment_data():
    hws = [1]
    reps = [1, 2]
    combined_df = None
    for hw in hws:
        for rep in reps:
            df = pd.read_csv("data/hw{}_r{}_experiment_data.csv".format(hw, rep), header=None)
            df.columns = ["cpu", "memory", "batch_size", "model_name", "quant_scheme", "accuracy", "inference_time",
                          "model_size_mb"]
            df["hardware"] = hw
            df["repetition"] = rep
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat((combined_df, df))
    return combined_df


if __name__ == "__main__":
    df = load_experiment_data()
    model = regression_without_interaction(df)
    # model = regression_with_pairwise_interactions(df)
    y_pred, y_true = get_true_and_pred(df, model)
    visualize_prediction_errors(y_pred, y_true)

