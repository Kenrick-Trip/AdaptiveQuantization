import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

pd.set_option('display.max_columns', 500)


def main():
    df = pd.read_csv("experiment_data.csv", header=None)
    df.columns = ["cpu", "memory", "batch_size", "model_name",
                  "quant_scheme", "accuracy", "inference_time", "model_size_mb"]
    fitted = ols(
        "inference_time ~ C(quant_scheme) + C(model_name) + C(cpu) + C(cpu)*C(model_name)*C(quant_scheme)",
        data=df).fit()

    table = sm.stats.anova_lm(fitted)
    print(table)


if __name__ == "__main__":
    main()
