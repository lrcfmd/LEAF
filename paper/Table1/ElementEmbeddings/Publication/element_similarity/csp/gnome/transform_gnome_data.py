import pandas as pd
df = pd.read_csv('GNoME_element_subs.csv', delimiter=';')

df['pairs'] = [' '.join(sorted([a,b])) for a, b in zip(df['1'],df['2'])]
df['sub'] = df['SUB'].apply(lambda x: float(x.replace(",", ".")))
df = df[['pairs', 'sub']]
df.to_csv('tmp.csv', index=False)
