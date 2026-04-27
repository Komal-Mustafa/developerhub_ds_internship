#!/usr/bin/env python
# Task 1: Exploring and Visualizing the Iris Dataset
# DevelopersHub Corporation - Data Science &amp; Analytics Internship
# ============================================================
# INTRODUCTION
# ============================================================
# The Iris dataset is one of the most famous datasets in machine learning.
# It contains measurements of 150 iris flowers from 3 different species:
# - Iris Setosa, Iris Versicolor, Iris Virginica
# Each flower has 4 features: sepal length, sepal width, petal length, petal width (in cm)
# GOAL: Explore, summarize, and visualize this dataset to understand patterns.
# ============================================================
# IMPORTING LIBRARIES           
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
# ============================================================
#LOADING THE DATA 
# ============================================================
# Load the Iris dataset from a CSV file
# Set visual style
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['figure.figsize'] = (10, 6)
# ============================================================
# STEP 1: LOAD THE DATASET
# ============================================================
print("=" * 60)
print("STEP 1: Loading the Iris Dataset")
print("=" * 60)
# Load using seaborn's built-in dataset
df = sns.load_dataset('iris')
print("Dataset loaded successfully!")
print(f"Source: seaborn built-in (originally UCI ML Repository)")
# ============================================================
# STEP 2: UNDERSTAND THE DATASET STRUCTURE
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Dataset Structure")
print("=" * 60)
print(f"\nShape (rows, columns): {df.shape}")
print(f"\nColumn Names: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nClass Distribution:")
print(df['species'].value_counts())
# ============================================================
# STEP 3: VISUALIZE THE DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Creating Visualizations")
print("=" * 60)
# --- Plot 1: Scatter Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Scatter Plots - Iris Dataset', fontsize=16, fontweight='bold')
sns.scatterplot(ax=axes[0], data=df, x='sepal_length', y='sepal_width',
 hue='species', s=80, alpha=0.8)
axes[0].set_title('Sepal Length vs Sepal Width')
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
sns.scatterplot(ax=axes[1], data=df, x='petal_length', y='petal_width',
 hue='species', s=80, alpha=0.8)
axes[1].set_title('Petal Length vs Petal Width')
axes[1].set_xlabel('Petal Length (cm)')
axes[1].set_ylabel('Petal Width (cm)')
plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("■ Scatter plots saved!")
# --- Plot 2: Histograms ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Feature Distributions - Iris Dataset', fontsize=16, fontweight='bold')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
for i, (feature, color) in enumerate(zip(features, colors)):
 row, col = i // 2, i % 2
 for species in df['species'].unique():
  axes[row][col].hist(df[df['species'] == species][feature],
  alpha=0.6, label=species, bins=15)
 axes[row][col].set_title(f'{feature.replace("_", " ").title()}')
 axes[row][col].set_xlabel('cm')
 axes[row][col].set_ylabel('Frequency')
 axes[row][col].legend()
plt.tight_layout()
plt.savefig('histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("■ Histograms saved!")
# --- Plot 3: Box Plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Box Plots - Outlier Detection', fontsize=16, fontweight='bold')
for i, feature in enumerate(features):
 row, col = i // 2, i % 2
 sns.boxplot(ax=axes[row][col], data=df, x='species', y=feature, palette='Set2')
 axes[row][col].set_title(f'{feature.replace("_", " ").title()}')
 axes[row][col].set_xlabel('Species')
 axes[row][col].set_ylabel('cm')
plt.tight_layout()
plt.savefig('box_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("■ Box plots saved!")
# --- Plot 4: Correlation Heatmap (Bonus) ---
plt.figure(figsize=(8, 6))
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
 square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("■ Correlation heatmap saved!")
# --- Plot 5: Pair Plot (Bonus) ---
pair_plot = sns.pairplot(df, hue='species', diag_kind='kde', height=2.5)
pair_plot.fig.suptitle('Pair Plot - All Feature Combinations', y=1.02,
 fontsize=14, fontweight='bold')
plt.savefig('pair_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("■ Pair plot saved!")
# ============================================================
# CONCLUSION
# ============================================================
print("\n" + "=" * 60)
print("CONCLUSION &amp; KEY INSIGHTS")
print("=" * 60)
print("""
1. DATASET OVERVIEW:
 - 150 samples, 4 numeric features, 1 target (species)
 - No missing values — clean dataset, ready to use
 - 3 species: setosa, versicolor, virginica (50 each)
2. SCATTER PLOT INSIGHTS:
 - Petal features (length &amp; width) clearly separate species
 - Setosa is easily distinguishable from others
 - Versicolor and Virginica overlap slightly in sepal measurements
3. HISTOGRAM INSIGHTS:
 - Sepal width is roughly normally distributed
 - Petal length &amp; width are bimodal — setosa is very different
 - Feature distributions confirm setosa is the most distinct species
4. BOX PLOT INSIGHTS:
 - Setosa has the smallest petals with no outliers
 - Virginica has the largest petals overall
 - Some outliers visible in sepal width for all species
5. CORRELATION INSIGHTS:
 - Petal length and petal width are highly correlated (0.96)
 - Sepal length correlates positively with petal measurements
 - Sepal width has low or negative correlation with other features
Page 16
CONCLUSION: Petal measurements are the most powerful features for
distinguishing iris species. This dataset is ideal for classification tasks.
""")
