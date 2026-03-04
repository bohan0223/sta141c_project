"""
Abalone Dataset Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

file_path = r"C:\Users\User\Desktop\abalone\abalone.data"

cols = [
    "Sex", "Length", "Diameter", "Height",
    "Whole weight", "Shucked weight", "Viscera weight", "Shell weight",
    "Rings"
]

df = pd.read_csv(file_path, header=None, names=cols)

df["Age"] = df["Rings"] + 1.5

print("Loaded shape:", df.shape)
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    print(f"\n--- {col} ---")

    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()
    skewness = skew(df[col])

    print(f"Mean: {mean:.4f}")
    print(f"Median: {median:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Skewness: {skewness:.4f}")

    #IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"Outliers (IQR method): {len(outliers)}")

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    # Histogram + KDE
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram + KDE: {col}")

    # Boxplot
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot: {col}")

    plt.tight_layout()
    plt.show()

predictors = [
    "Length", "Diameter", "Height",
    "Whole weight", "Shucked weight",
    "Viscera weight", "Shell weight"
]

for col in predictors:
    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=df[col],
        y=df["Age"],
        lowess=True,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"}
    )
    plt.title(f"Age vs {col}")
    plt.show()

for col in ["Whole weight", "Shucked weight",
            "Viscera weight", "Shell weight"]:
    df[f"log_{col}"] = np.log(df[col] + 1e-6)

    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=df[f"log_{col}"],
        y=df["Age"],
        lowess=True,
        scatter_kws={"alpha": 0.3},
        line_kws={"color": "red"}
    )
    plt.title(f"Age vs log({col})")
    plt.show()

predictors = [
    "Length", "Diameter", "Height",
    "Whole weight", "Shucked weight",
    "Viscera weight", "Shell weight"
]

predictors = [
    "Length", "Diameter", "Height",
    "Whole weight", "Shucked weight",
    "Viscera weight", "Shell weight"
]

corr = df[predictors].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

X = df[predictors]
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif_data)