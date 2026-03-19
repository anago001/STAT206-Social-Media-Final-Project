import itertools
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

#VIF Check Function
def check_vif(df, x_vars):
    X = df[x_vars].copy()
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]

    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_data
