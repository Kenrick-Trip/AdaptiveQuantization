import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.max_columns', 500)


def regression_without_interaction():
    df = pd.read_csv("experiment_data.csv", header=None)
    df.columns = ["cpu", "memory", "batch_size", "model_name",
                  "quant_scheme", "accuracy", "inference_time", "model_size_mb"]
    model = ols(
        "inference_time ~ C(cpu) + C(memory) + C(batch_size) + C(model_name) + C(quant_scheme)",
        data=df).fit()

    table = sm.stats.anova_lm(model)
    print(table)
    print(model.summary())


if __name__ == "__main__":
    regression_without_interaction()
