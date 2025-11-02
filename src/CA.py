import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import os


# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])

df['target'] = df['target'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Select only the numerical attributes
numerical_df = df.drop('target', axis=1)

# Calculate the correlation matrix
correlation_matrix = numerical_df.corr()

# Save the correlation matrix to a CSV file
output_path = os.path.join("..", "dist", "correlations.csv")

# Save the correlation matrix to a CSV file in the dist folder
correlation_matrix.to_csv(output_path, header=False, index=False)
# Find the pair of features with the minimum and maximum absolute correlation
min_corr_pair = correlation_matrix.stack().nsmallest(1).index[0]
max_corr_pair = correlation_matrix.stack().nlargest(1).index[0]

print("Pair of features with minimum absolute correlation:", min_corr_pair)
print("Pair of features with maximum absolute correlation:", max_corr_pair)