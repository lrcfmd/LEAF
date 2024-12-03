import pandas as pd
import sys

f1, f2, name= sys.argv[1:4]

def transform(f):
    df = pd.read_csv(f)
    df.index = df['element']
    df = df.T
    df = df.iloc[1:]
    return df

df1 = transform(f1)
df2 = transform(f2)

df = pd.concat([df1, df2])
df = df.T
df = df.reset_index()

df.to_csv(f'{name}.csv', index=False)
