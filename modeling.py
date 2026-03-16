import itertools
import statsmodels.api as sm
import pandas as pd

#Create Dummy Variable Function
def dummies(df, column, drop_first=True):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df_copy = df.copy()

    if df_copy[column].dtype == "bool":
        df_copy[column] = df_copy[column].astype(int)

    elif df_copy[column].dtype == "object" or str(df_copy[column].dtype) == "category":
        dummies = pd.get_dummies(df_copy[column], prefix=column, drop_first=drop_first)
        dummies = dummies.astype(int) 
        df_copy = pd.concat([df_copy.drop(columns=[column]), dummies], axis=1)

    return df_copy


#Regression Model Function
def regression(df, y_var, x_vars):
    df = df.copy()
    
    for col in x_vars + [y_var]:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)
    
    X = df[x_vars]
    X = sm.add_constant(X) 
    y = df[y_var]
    
    model = sm.OLS(y, X).fit()
    
    print(model.summary())
    return model
