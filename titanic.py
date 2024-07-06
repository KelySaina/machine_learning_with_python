import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Using seaborn's built-in dataset
df = sns.load_dataset('titanic')

# Display the first few rows
print(df.head())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the distribution of the target variable
sns.countplot(x='survived', data=df)
plt.show()

# Fill missing age values with the median age
df['age'].fillna(df['age'].median(), inplace=True)

# Drop rows where 'embarked' is missing
df.dropna(subset=['embarked'], inplace=True)

# Drop columns that are not needed
df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1, inplace=True)

# Check again for missing values
print(df.isnull().sum())

# Convert categorical columns to numeric
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop rows with missing 'age' values
df.dropna(subset=['age'], inplace=True)

# Define features and target variable
X = df.drop(['survived'], axis=1)
y = df['survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Confusion matrix
print(confusion_matrix(y_test, y_pred))

# Classification report
print(classification_report(y_test, y_pred))

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy in percentage with two decimal points
print(f'Accuracy: {accuracy:.2%}')