import pandas as pd

df = pd.read_csv('leaf_neighbour_sum.csv')
nf = pd.read_csv('nleaf_neighbour_sum.csv')
df = df.drop(columns=['X'])
nf = nf.drop(columns=['X'])

for col in df.columns:
    df[col] = df[col].values / nf[col].values

df = df.fillna(0)
df = df.transpose()
df = df.reset_index()
df = df.rename(columns={'index':'elements'})

df.to_csv('LEAF_NEIGH.csv', index=False)
