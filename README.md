# Using KNNImputer for Advanced Data Imputation

## Introduction
While traditional imputation techniques like mean, median, and mode are widely used for handling missing data, machine learning-based imputation methods provide an alternative approach. In this tutorial, we will explore how to use the scikit-learn's `KNNImputer` to leverage machine learning for data imputation.

### Why Use KNNImputer for Imputation?
KNNImputer offers several advantages over traditional imputation techniques:
1. **Contextual Imputation:** KNNImputer uses the K-nearest neighbors algorithm to impute missing values, considering the relationships between data points. This results in more context-aware imputations.
2. **Adaptability:** KNNImputer can be used for both numerical and categorical data, making it versatile for various types of datasets.
3. **Data Distribution Preservation:** Imputations made by KNNImputer tend to preserve the statistical characteristics of the data, which can be important for downstream analysis.

## Usage Guide

### Step 1: Import Necessary Libraries
Start by importing the required libraries, including `KNNImputer` from scikit-learn:

```python
from sklearn.impute import KNNImputer
import numpy as np```

### Step 2: Prepare Your Data
Create or load your dataset with missing values:

```Create a sample dataset with missing values
data = np.array([[1, 2, 3],
                 [4, 5, np.nan],
                 [7, 8, 9],
                 [np.nan, 11, 12]])```
				 
### Step 3: Initialize KNNImputer
Initialize the KNNImputer object with the desired number of neighbors (k) and other optional parameters:

``` Initialize the KNNImputer with 2 neighbors
imputer = KNNImputer(n_neighbors=2)```


