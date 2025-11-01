import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

df["total score"] = df['math score'] + df['reading score'] + df['writing score']

df.to_csv('StudentsPerformance.csv', index=False)
