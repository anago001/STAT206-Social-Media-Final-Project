import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def summarize_variable(df, column):
    summary = df[column].describe()
    print("Summary Statistics:")
    print(summary)
    
    plt.figure(figsize=(8, 5))
    
    if df[column].dtype == "bool":
        sns.countplot(x=df[column])
        plt.title(f"Distribution of {column} (True/False)")
        plt.xlabel(column)
        plt.ylabel("Count")
    elif df[column].dtype == "object":
        sns.countplot(x=df[column])
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
    else:
        sns.histplot(df[column], bins=20, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
    
    plt.show()

#Calculate Correlation Function
def correlate(df, col1, col2, method="pearson", plot=True):
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns '{col1}' or '{col2}' not found in DataFrame.")

    series1 = df[col1]
    series2 = df[col2]
    if series1.dtype == "bool":
        series1 = series1.astype(int)
    if series2.dtype == "bool":
        series2 = series2.astype(int)
    
    corr_value = series1.corr(series2, method=method)
    print(f"{method.capitalize()} correlation between '{col1}' and '{col2}': {corr_value:.3f}")
    
    if plot:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x=series1, y=series2)
        sns.regplot(x=series1, y=series2, scatter=False, color="red")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f"{method.capitalize()} correlation: {corr_value:.3f}")
        plt.show()
    
    return corr_value

#Compare Scale and Categorical Data Function
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway

def compare(df, scale_col, cat_col, horizontal=False):
    if scale_col not in df.columns or cat_col not in df.columns:
        raise ValueError(f"Columns '{scale_col}' or '{cat_col}' not found in DataFrame.")
    
    data = df[[scale_col, cat_col]].dropna()
    data[cat_col] = data[cat_col].astype("category")
    
    groups = data[cat_col].cat.categories
    num_groups = len(groups)
    
    print(f"Comparing '{scale_col}' across {num_groups} groups of '{cat_col}': {list(groups)}")
    
    plt.figure(figsize=(10,6))
    
    if horizontal:
        sns.boxplot(y=cat_col, x=scale_col, data=data)
        plt.ylabel(cat_col)
        plt.xlabel(scale_col)
    else:
        sns.boxplot(x=cat_col, y=scale_col, data=data)
        plt.xlabel(cat_col)
        plt.ylabel(scale_col)
        plt.xticks(rotation=45, ha="right") 
    
    plt.title(f"{scale_col} by {cat_col}")
    plt.tight_layout()
    plt.show()
    
    if num_groups == 2:
        group1 = data[data[cat_col]==groups[0]][scale_col]
        group2 = data[data[cat_col]==groups[1]][scale_col]
        stat, pval = ttest_ind(group1, group2, equal_var=False)
        print(f"T-test (Welch) result: t = {stat:.3f}, p = {pval:.4f}")
    elif num_groups > 2:
        samples = [data[data[cat_col]==g][scale_col] for g in groups]
        stat, pval = f_oneway(*samples)
        print(f"One-way ANOVA result: F = {stat:.3f}, p = {pval:.4f}")
    else:
        print("Categorical variable must have at least 2 groups.")
        return None
    
    return stat, pval