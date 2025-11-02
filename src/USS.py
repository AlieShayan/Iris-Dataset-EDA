import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import os


iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])

df['target'] = df['target'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

grouped_df = df.groupby('target')

statistics = []
for name, group in grouped_df:
    sepal_width = group['sepal width (cm)'].dropna()
    missing_values = group['sepal width (cm)'].isnull().sum()
    min_value = sepal_width.min()
    q1 = sepal_width.quantile(0.25)
    median = sepal_width.median()
    q3 = sepal_width.quantile(0.75)
    p95 = sepal_width.quantile(0.95)
    max_value = sepal_width.max()
    mean = sepal_width.mean()
    range_value = max_value - min_value
    iqr = q3 - q1
    std = sepal_width.std()
    std_pop = sepal_width.std(ddof=0)
    mad = np.median(np.abs(sepal_width - median))

    statistics.append([name, missing_values, min_value, q1, median, q3, p95, max_value, mean, range_value, iqr, std, std_pop, mad])

# Create a DataFrame from the statistics
statistics_df = pd.DataFrame(statistics, columns=['label', 'missing', 'min', 'q1',
                                                  'med', 'q3', 'p95', 'max', 'mean',
                                                  'range', 'iqr', 'std', 'std_pop', 'mad'])

output_path = os.path.join("..", "dist", "statistics.csv")

# Save the correlation matrix to a CSV file in the dist folder
statistics_df.to_csv(output_path, header=True, index=False)