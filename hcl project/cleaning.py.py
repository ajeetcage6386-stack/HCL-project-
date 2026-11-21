# Student Marks Prediction Dataset - Data Cleaning
# Input : student_marks_prediction_uncleaned_858.csv
# Output: student_marks_prediction_cleaned.csv

import pandas as pd
import numpy as np

# 1. Load the uncleaned CSV file
df = pd.read_csv("student_marks_prediction_uncleaned_858.csv")

print("Original shape:", df.shape)
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 2. Remove duplicate rows
df = df.drop_duplicates().copy()

# 3. Clean text columns: remove leading/trailing spaces
text_cols = [
    "Name", "Gender", "Class", "City", "Tutoring",
    "Internet_Access", "Extracurricular_Activities"
]

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# 4. Standardize categorical text
for col in [
    "Gender", "City", "Tutoring",
    "Internet_Access", "Extracurricular_Activities"
]:
    df[col] = df[col].str.title()

df["Name"] = df["Name"].str.title()

# Normalize class values such as "10.0" to "10"
df["Class"] = df["Class"].str.replace(r"\.0$", "", regex=True)

# 5. Convert numeric columns to numeric data types.
# Invalid text (e.g. "five", "Absent", "one hundred") becomes NaN.
numeric_cols = [
    "Age", "Study_Hours_Per_Day", "Attendance_Percentage",
    "Previous_Score", "Sleep_Hours", "Final_Marks"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Replace values outside reasonable ranges with NaN
valid_ranges = {
    "Age": (10, 25),
    "Study_Hours_Per_Day": (0, 24),
    "Attendance_Percentage": (0, 100),
    "Previous_Score": (0, 100),
    "Sleep_Hours": (0, 24),
    "Final_Marks": (0, 100)
}

for col, (minimum, maximum) in valid_ranges.items():
    invalid = ~df[col].between(minimum, maximum)
    df.loc[invalid, col] = np.nan

# 7. Keep only valid category values
valid_categories = {
    "Gender": ["Male", "Female"],
    "Class": ["9", "10", "11", "12"],
    "Tutoring": ["Yes", "No"],
    "Internet_Access": ["Yes", "No"],
    "Extracurricular_Activities": ["Yes", "No"]
}

for col, valid_values in valid_categories.items():
    df.loc[~df[col].isin(valid_values), col] = pd.NA

# 8. Fill missing numeric values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# 9. Fill missing categorical values with mode
categorical_cols = [
    "Gender", "Class", "City", "Tutoring",
    "Internet_Access", "Extracurricular_Activities"
]

for col in categorical_cols:
    mode_values = df[col].mode(dropna=True)
    fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
    df[col] = df[col].fillna(fill_value)

# 10. Fill missing names
df["Name"] = df["Name"].fillna("Unknown Student")

# 11. Set appropriate data types
df["Age"] = df["Age"].round().astype("int64")
df["Class"] = df["Class"].astype("string")

# 12. Save the cleaned dataset in a new CSV file
df.to_csv("student_marks_prediction_cleaned.csv", index=False, encoding="utf-8-sig")

# 13. Verify the cleaned data
print("\nCleaned shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nTotal missing values:", df.isnull().sum().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nCleaned dataset saved as student_marks_prediction_cleaned.csv")
