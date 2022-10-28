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


def regression_with_pair_interactions(data):
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


if __name__ == "__main__":
    df = pd.read_csv("experiment_data.csv", header=None)
    df.columns = ["cpu", "memory", "batch_size", "model_name",
                  "quant_scheme", "accuracy", "inference_time", "model_size_mb"]
    regression_without_interaction(df)
    # regression_with_pair_interactions(df)
