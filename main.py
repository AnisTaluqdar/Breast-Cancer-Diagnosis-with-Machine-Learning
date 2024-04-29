

# %%
# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython import display
display.set_matplotlib_formats('svg')
# %%
# Load the dataset and explore its characteristics.
df = pd.read_csv('/home/anis/Documents/Projects/breast_cancer_wisconsin.csv')
df
# %%
# Handle missing values (if any).
print(df.isnull().sum().sum())

print(df.isnull().sum())

# remove null value
df.dropna(axis=1, inplace=True)

# check null value again
print(df.isnull().sum().sum())
# %%
# Data description
df.describe()
# %%
# data information
df.info()
# %%
df['diagnosis']
# %%
# one-hot encoding

df = pd.get_dummies(df, columns=['diagnosis'], drop_first=True)
# %%
df
# %%

# Normalize features (e.g., Min-Max scaling or Z-score normalization).

# %%
# Split the data into training and testing sets.