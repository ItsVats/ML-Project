import pandas as pd
from pathlib import Path

p = Path(r'notebook\data\StudentsPerformance.csv')

df = pd.read_csv(p)

df.drop(columns=['total score'],inplace=True)

df.to_csv(p,index=False)