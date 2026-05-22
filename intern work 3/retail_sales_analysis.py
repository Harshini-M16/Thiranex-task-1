# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Create simple sales data
data = {
    "Product": ["Shampoo", "Soap", "Oil", "Paste", "Brush"],
    "Sales": [120, 90, 150, 80, 60]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Show dataset
print("Sales Data")
print(df)

# Find total sales
total = df["Sales"].sum()
print("\nTotal Sales:", total)

# Find highest selling product
best = df.loc[df["Sales"].idxmax()]
print("\nBest Selling Product:")
print(best)

# Plot bar chart
plt.bar(df["Product"], df["Sales"])
plt.title("Product Sales")
plt.xlabel("Products")
plt.ylabel("Sales")
plt.show()