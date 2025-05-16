import pandas as pd
df= pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

#Display the first few rows of the dataset using .head() to inspect the data.
print(df.head())

#Datatypes
print("Data types of each column: ", df.dtypes)

#Missing columns
print("Missing values for each column:", df.isnull().sum())

#Filling missing values
df_filled_0 = df.fillna(0) #Fill missing no. with 0
df_filled_mean =df.fillna(df.mean(numeric_only=True))
df_filled_ffill=df.fillna(method='ffill')#Forward fill
df_filled_bfill=df.fillna(method='bfill')#Backward fill
#Filling in specific columns
df['species'] = df['species'].fillna('unknown')#One column
df['sepal_width']=df['sepal_width'].fillna(df['sepal_width'].median())

#Dropping missing values
df_dropped_rows= df.dropna() #Drop all rows with missing marks

#Drop columns with any missing values:
df_dropped_cols = df.dropna(axis=1)

#Drop rows with missing values in specific columns:
df_dropped_subset = df.dropna(subset=['species', 'sepal_width'])

# Print results to see the effect of each operation
print("\nFilled with 0:\n", df_filled_0.isnull().sum())
print("\nFilled with Mean:\n", df_filled_mean.isnull().sum())
print("\nFilled with ffill:\n", df_filled_ffill.isnull().sum())
print("\nFilled with bfill:\n", df_filled_bfill.isnull().sum())

print("\nDropped Rows:\n", df_dropped_rows.isnull().sum())
print("\nDropped Columns:\n", df_dropped_cols.isnull().sum())
print("\nDropped Subset:\n", df_dropped_subset.isnull().sum())

# Compute the basic statistics of the numerical columns 
numerical_statistics = df.describe()
print("Basic statistics of numerical columns: ", numerical_statistics)

# groupings on a categorical column
grouped_statistics=df.groupby('species').mean()
grouped_statistics=df.groupby('species').median()
grouped_statistics=df.groupby('species').std()

# Print the statistics
print(numerical_statistics)

# 1.Identify any patterns or interesting findings from your analysis.
# No Missing values in the dataset.
# Species difference in terms of their width, height, sizes.

#Create visualizations:
#Line chart showing trends over time (for example, a time-series of sales data).
import matplotlib.pyplot as plt
import seaborn as sns

#Line chart showing trends over time
plt.figure(figsize=(8,4))
plt.plot(df.index, df['sepal_length'], color='purple', label='Sepal Length')
plt.title('Sepal Length Across Samples')
plt.xlabel('Samples')
plt.ylabel('Sepal Length(cm)')
plt.legend()
plt.tight_layout()
plt.show()

#Bar chart showing the comparison of a numerical value across categories (e.g., average petal length per species).
plt.figure(figsize=(6,4))
sns.barplot(x='species', y='petal_length', data=df, estimator='mean', ci=None)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length')
plt.tight_layout()
plt.show()

#Histogram of a numerical column to understand its distribution.
plt.figure(figsize=(6,4))
sns.histplot(df['sepal_width'], bins=15, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#Scatter plot to visualize the relationship between two numerical columns (e.g., sepal length vs. petal length).
plt.figure(figsize=(6,4))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length(cm)')
plt.ylabel('Petal Length(cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()