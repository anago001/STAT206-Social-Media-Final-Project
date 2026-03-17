import itertools
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Create Dummy Variable Function
def dummies(df, column, drop_first=True):
    """
    Converts a categorical or boolean column into dummy variables and
    appends them to the DataFrame, dropping the original column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the column to encode.
    drop_first : bool, optional
        Whether to drop the first dummy category to avoid multicollinearity.
        Default is True.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original column replaced by dummy variables.

    Raises
    ------
    ValueError
        If the column is not found in the DataFrame.
    """

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
    """
    Fits an OLS (Ordinary Least Squares) linear regression model and prints
    the full summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing both the dependent and independent variables.
    y_var : str
        Name of the dependent (target) variable column.
    x_vars : list of str
        List of column names to use as independent (predictor) variables.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted OLS regression model object.
    """
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

def check_vif(df, x_vars):
    """
    Computes the Variance Inflation Factor (VIF) for each predictor variable
    to detect multicollinearity.

    VIF = 1 means no correlation with other predictors.
    VIF between 1 and 5 is generally acceptable.
    VIF above 10 indicates high multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the predictor variables.
    x_vars : list of str
        List of column names for which to compute VIF.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'Variable' and 'VIF', sorted by VIF descending.
    """
    X = df[x_vars].copy()
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]

    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_data