import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("Advertising.csv")

# Display first 5 rows
print(df.head())

# Dataset Information
print(df.info())

# Check Null Values
print(df.isnull().sum())

# Data Visualization
sns.pairplot(df)
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("---------------------")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Comparing Actual vs Predicted
comparison = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": predictions
})

print("\nActual vs Predicted")
print(comparison.head())

# Scatter Plot
plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
