import pandas as pd
import sys

df = pd.read_csv('LEAF_NEIGH_MAGPIE.csv')

print(df.head())
print(df.shape)

#df = df.T
#df = df.reset_index()

#df.to_csv('LEAF_2_200.csv', index=False)
