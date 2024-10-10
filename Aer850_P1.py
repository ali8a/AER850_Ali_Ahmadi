#importing required libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
'''STEP 1'''
#Reading The CSV file
df = pd.read_csv("Project_1_Data.csv")

#looking at the data
print(df.head())

'''STEP 2'''
#Defining each column 
x = df['X']
y = df['Y']
z = df['Z']
s = df['Step']

# Create a 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot X, Y, Z with respect to the step column
sc = ax.scatter(x, y, z, c=s, cmap='viridis', marker='o')

# Label the axes and title of the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D plot of X, Y, Z')

# Add a color bar to indicate the values of 'Step'
plt.colorbar(sc, label='Step')

# Show the plot
plt.show()

'''STEP 3'''

# Correlation Analysis
# Create a new DataFrame with the features and the target variable
correlation_df = df[['X', 'Y', 'Z','Step']]

# Check if correlation_df is a DataFrame
print(f"\nType of correlation_df: {type(correlation_df)}")  # Should output <class 'pandas.core.frame.DataFrame'>

# Calculate the Pearson correlation matrix
correlation_matrix = correlation_df.corr(method='pearson')

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create a heatmap for the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

'''STEP 4'''
