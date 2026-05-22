# ================================
# Exploratory Data Analysis (EDA)
# Basic and Easy Python Project
# ================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
# Replace 'data.csv' with your dataset file name
df = pd.read_csv('data.csv')

# -------------------------------
# Display first 5 rows
# -------------------------------
print("\nFirst 5 Rows:")
print(df.head())

# -------------------------------
# Dataset information
# -------------------------------
print("\nDataset Information:")
print(df.info())

# -------------------------------
# Statistical Summary
# -------------------------------
print("\nStatistical Summary:")
print(df.describe())

# -------------------------------
# Check Missing Values
# -------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Correlation between columns
# -------------------------------
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# -------------------------------
# Histogram
# -------------------------------
df.hist(figsize=(10, 8))
plt.suptitle("Histogram of Dataset")
plt.show()

# -------------------------------
# Box Plot
# -------------------------------
df.plot(kind='box', figsize=(10, 6))
plt.title("Box Plot")
plt.show()

# -------------------------------
# Correlation Heatmap
# -------------------------------
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')

plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# Final Insights
# -------------------------------
print("\nEDA Completed Successfully!")
print("Insights:")
print("- Checked dataset structure")
print("- Found missing values")
print("- Analyzed statistics")
print("- Visualized data using charts")
print("- Identified correlations")