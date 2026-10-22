import pandas as pd

df=pd.read_csv("employees.csv")

# print(df)

"""
print(df.isnull())

print(df.isnull().sum()) #But imagine a dataset with 100,000 rows. Checking every True manually would be difficult.

print(df.isnull().sum().sum()) #This will give you the total number of missing values in the entire dataset.

print(df.duplicated()) #checks wheather a row has appeared before or not

print(df.duplicated().sum()) #This will give you the total number of duplicate rows in the entire dataset.

print(df[df.duplicated()]) #Display the Actual Duplicate Rows

print(df[df["Age"]>100]) #This will give you the rows where the Age is greater than 100.

print((df["Age"]>100).sum())

"""

missing=df.isnull().sum().sum()
duplicates=df.duplicated().sum()
#suspicious_ages=(df["Age"] > 100).sum()

print("missing values:",missing)

print("duplicate rows:",duplicates)

#print("suspicious ages:",suspicious_ages)
if "Age" in df.columns:
    suspicious_ages=(df["Age"] > 100).sum()
    print("suspicious ages:",suspicious_ages)
else:
    print("Age column not found in the dataset.")

