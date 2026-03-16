import pandas as pd
import numpy as np

def clean_data(df):
    numeric_cols = df.select_dtypes(include=np.number)

    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((numeric_cols < (Q1 - 1.5 * IQR)) |
                (numeric_cols > (Q3 + 1.5 * IQR)))

    outlier_rows = df[outliers.any(axis=1)]

    df_clean = df[~outliers.any(axis=1)]

    return df_clean