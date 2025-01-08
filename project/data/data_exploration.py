import pandas as pd

df = pd.read_csv('detections.csv')

print(df.head(10))
print(df.info())