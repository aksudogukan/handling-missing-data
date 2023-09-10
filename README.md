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
import numpy as np
```

### Step 2: Prepare Your Data
Create or load your dataset with missing values:

```python
# Create a sample dataset with missing values
data = np.array([[1, 2, 3],
                 [4, 5, np.nan],
                 [7, 8, 9],
                 [np.nan, 11, 12]])
```
Also, there is an example on 'fetch_california_housing' dataset in the KNNImputer_Handling Missing Values.ipynb.
				 
### Step 3: Initialize KNNImputer
Initialize the KNNImputer object with the desired number of neighbors (k) and other optional parameters:

```python
# Initialize the KNNImputer with 2 neighbors
imputer = KNNImputer(n_neighbors=2)
```

### Step 4: Fit and Transform
Fit the imputer on your dataset and transform it to fill in the missing values:

```python
# Fit and transform the imputer on the data
imputed_data = imputer.fit_transform(data)
```

imputed_data now contains your original dataset with the missing values filled in using the K-nearest neighbors algorithm.

### Step 5: View the Result
Print the imputed dataset to observe the changes:

```python
# Print the imputed dataset
print(imputed_data)
```

### Step 6: Customizing KNNImputer
You can customize the behavior of KNNImputer by modifying its parameters. For example, you can change the number of neighbors or the distance metric used for imputation:

```python
# Example: Using 3 neighbors and Manhattan distance
imputer = KNNImputer(n_neighbors=3, metric='manhattan')
imputed_data = imputer.fit_transform(data)
```

### Step 7: Handling Additional Parameters
KNNImputer provides additional parameters for advanced users, such as weights, missing_values, and more. Explore these options in the scikit-learn documentation for fine-tuning your imputation process.

### Conclusion
In this tutorial, you've learned how to use the KNNImputer in Python to impute missing values in your datasets. KNN imputation can provide more context-aware and accurate imputations compared to traditional methods. Remember to choose the appropriate number of neighbors and consider the nature of your data when using KNNImputer for imputation tasks. It's a valuable tool to have in your data preprocessing toolkit for handling missing data effectively.


