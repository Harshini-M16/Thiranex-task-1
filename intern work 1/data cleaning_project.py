# ======================================================
# DATA CLEANING AND VISUALIZATION PROJECT
# ======================================================

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# CREATE SAMPLE DATASET


data = {
    "Name": ["A", "B", "C", "D", "E", "F", "G", "A"],
    "Age": [23, 25, None, 22, 120, 24, 23, 23],
    "Salary": [50000, 54000, 58000, None, 1000000, 52000, 51000, 50000],
    "Department": ["HR", "IT", "Finance", "IT", "HR", None, "Finance", "HR"]
}

# CREATE DATAFRAME
df = pd.DataFrame(data)

# DISPLAY ORIGINAL DATA


print("\n========== ORIGINAL DATA ==========\n")
print(df)

# CHECK MISSING VALUES


print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# HANDLE MISSING VALUES


# Fill missing Age with average age
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Fill missing Salary with average salary
df["Salary"].fillna(df["Salary"].mean(), inplace=True)

# Fill missing Department with 'Unknown'
df["Department"].fillna("Unknown", inplace=True)


# REMOVE DUPLICATES


df.drop_duplicates(inplace=True)

print("\n========== AFTER REMOVING DUPLICATES ==========\n")
print(df)

# OUTLIER DETECTION


print("\n========== OUTLIER CHECK ==========\n")

# Find unusual age values
outliers = df[df["Age"] > 100]

print(outliers)

# Remove outliers
df = df[df["Age"] < 100]

# CLEANED DATA


print("\n========== CLEANED DATA ==========\n")
print(df)

# DATA VISUALIZATION


# STYLE
sns.set(style="whitegrid")

# AGE DISTRIBUTION


plt.figure(figsize=(6,4))

sns.histplot(df["Age"], bins=5)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()

# SALARY DISTRIBUTION


plt.figure(figsize=(6,4))

sns.boxplot(y=df["Salary"])

plt.title("Salary Boxplot")

plt.show()

# DEPARTMENT COUNT


plt.figure(figsize=(6,4))

sns.countplot(x="Department", data=df)

plt.title("Department Count")

plt.show()


# AGE VS SALARY


plt.figure(figsize=(6,4))

sns.scatterplot(x="Age", y="Salary", data=df)

plt.title("Age vs Salary")

plt.show()

# ======================================================
# FINAL MESSAGE
# ======================================================

print("\n===================================")
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("VISUALIZATION GENERATED")
print("===================================")