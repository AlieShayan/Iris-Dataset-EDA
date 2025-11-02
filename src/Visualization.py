import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
# Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df['species'] = pd.Categorical.from_codes(iris.target, categories=iris.target_names)

# Distribution of Label
sns.countplot(x='species', data=df, palette='Set2', hue='species', legend=False)
plt.title('Distribution of Iris Species')
plt.show()

# Histograms
def plot_histograms(df, features):
    plt.figure(figsize=(12, 5))
    for i, feature in enumerate(features, start=1):
        plt.subplot(1, 2, i)
        sns.histplot(df[feature], bins=35, kde=True, color='skyblue')
        plt.title(f'Histogram of {feature.replace("_", " ").capitalize()}')
    plt.tight_layout()
    plt.show()

plot_histograms(df, ['petal_length', 'sepal_width'])

# 3D Histogram
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(df['petal_length'], df['sepal_length'], bins=35, range=[[0, 7], [4, 8]])
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos, ypos, zpos = xpos.ravel(), ypos.ravel(), 0
dx = dy = 0.25 * np.ones_like(zpos)
dz = hist.ravel()
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Sepal Length')
ax.set_zlabel('Count')
ax.view_init(elev=45)

plt.title('3D Histogram of Petal Length and Sepal Length')
plt.show()

# Box Plots
def plot_boxplots(df, features):
    plt.figure(figsize=(12, 6))
    for i, feature in enumerate(features, start=1):
        plt.subplot(1, 2, i)
        sns.boxplot(x='species', y=feature, data=df, palette='Set2', hue='species', legend=False)
        plt.title(f'Box Plot of {feature.replace("_", " ").capitalize()} by Species')
    plt.tight_layout()
    plt.show()

plot_boxplots(df, ['petal_length', 'sepal_width'])

# 2D Box Plot with Outliers
def plot_2d_box_with_outliers(df, x_feature, y_feature):
    def find_outliers(series):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        return series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]

    def calc_min_max(series, is_min=True):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        return max(q3 - 1.5 * iqr, min(series)) if is_min else min(q3 + 1.5 * iqr, max(series))

    x_stats, y_stats = df[x_feature].describe(), df[y_feature].describe()
    outliers = df[(df[x_feature].isin(find_outliers(df[x_feature]))) |
                  (df[y_feature].isin(find_outliers(df[y_feature])))]
    x_stats['min'], x_stats['max'] = calc_min_max(df[x_feature], True), calc_min_max(df[x_feature], False)
    y_stats['min'], y_stats['max'] = calc_min_max(df[y_feature], True), calc_min_max(df[y_feature], False)

    plt.figure(figsize=(10, 6))
    color = sns.color_palette("Set2", 1)[0]
    plt.gca().add_patch(Rectangle((x_stats['25%'], y_stats['25%']),
                                  x_stats['75%'] - x_stats['25%'],
                                  y_stats['75%'] - y_stats['25%'],
                                  fill=True, color=color, alpha=0.5))
    plt.plot([x_stats['min'], x_stats['max']], [y_stats['50%'], y_stats['50%']], color=color)
    plt.plot([x_stats['50%'], x_stats['50%']], [y_stats['min'], y_stats['max']], color=color)
    plt.plot([x_stats['min'], x_stats['min']], [y_stats['25%'], y_stats['75%']], color=color)
    plt.plot([x_stats['max'], x_stats['max']], [y_stats['25%'], y_stats['75%']], color=color)
    plt.plot([x_stats['25%'], x_stats['75%']], [y_stats['min'], y_stats['min']], color=color)
    plt.plot([x_stats['25%'], x_stats['75%']], [y_stats['max'], y_stats['max']], color=color)
    plt.scatter(outliers[x_feature], outliers[y_feature], color='black', marker='o', label='Outliers')
    plt.xlabel(f'{x_feature.replace("_", " ").capitalize()} (cm)')
    plt.ylabel(f'{y_feature.replace("_", " ").capitalize()} (cm)')
    plt.legend()
    plt.title(f"2D Boxplot for {x_feature.replace('_', ' ').capitalize()} and {y_feature.replace('_', ' ').capitalize()}")
    plt.show()

plot_2d_box_with_outliers(df, 'petal_length', 'sepal_width')

# Quantile Plots
def quantile_plot(series):
    sorted_data = series.sort_values().reset_index(drop=True)
    N = len(sorted_data)
    quantiles = [(i - 0.5) / N for i in range(1, N + 1)]
    plt.scatter(quantiles, sorted_data, color='orange', edgecolor='black')
    plt.xlabel("Quantiles")
    plt.ylabel("Value")
    plt.title(f"Quantile Plot for {series.name.replace('_', ' ').capitalize()}")
    plt.show()

quantile_plot(df['sepal_width'])
quantile_plot(df['petal_length'])

# Scatter Plots
def plot_feature_pairs(df, features):
    pair_indices = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    plt.figure(figsize=(18, 10))
    for idx, (i, j) in enumerate(pair_indices, start=1):
        plt.subplot(2, 3, idx)
        sns.scatterplot(x=features[i], y=features[j], hue='species', data=df, palette='Set2', legend=False)
        plt.title(f'{features[i].capitalize()} vs {features[j].capitalize()}')
    plt.tight_layout()
    plt.show()

plot_feature_pairs(df, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# 3D Scatter Plot
fig = plt.figure(figsize=(16, 8))
colors = {'setosa': 'r', 'versicolor': 'g', 'virginica': 'b'}
ax1, ax2 = fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')

for species, color in colors.items():
    subset = df[df['species'] == species]
    for ax in [ax1, ax2]:
        ax.scatter(subset['sepal_length'], subset['sepal_width'], subset['petal_length'], color=color, label=species)
ax1.set(title="3D Scatter Plot - View 1", xlabel='Sepal Length', ylabel='Sepal Width', zlabel='Petal Length')
ax1.view_init(elev=15, azim=40)
ax2.set(title="3D Scatter Plot - View 2", xlabel='Sepal Length', ylabel='Sepal Width', zlabel='Petal Length')
ax2.view_init(elev=70, azim=40)
ax1.legend()
plt.show()

# Probability Distributions
plt.figure(figsize=(8, 6))
for species, color in colors.items():
    subset = df[df['species'] == species]
    density = gaussian_kde(subset['petal_length'])
    xs = np.linspace(subset['petal_length'].min(), subset['petal_length'].max(), 200)
    plt.plot(xs, density(xs), label=species, color=color)
plt.title('Probability Density of Petal Length by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Density')
plt.legend()
plt.show()