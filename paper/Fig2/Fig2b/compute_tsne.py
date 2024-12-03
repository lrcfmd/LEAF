# sklearn v 1.5.1
import sys
import matplotlib.pyplot as plt
from pymatgen.core import Composition

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px


def tsne(X, rs=13):
    X_embedded = TSNE(random_state=rs, n_components=2, learning_rate='auto',init='pca', perplexity=3).fit_transform(X)
    return X_embedded

#CRYSTALS = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal', 'triclinic', 'trigonal']

def get_st(df):
    vals = df['structure_type'].value_counts()
    df['st_entries'] = vals
    df = df.sort_values('st_entries', ascending=False)
    #vals = vals[vals>=1253]
    vals = vals[vals<1000]
    vals = vals[vals>500]
    st = list(vals.index)
    df = df[df['structure_type'].isin(st)]
    df['size'] = [2 for i in range(df.shape[0])]
    return df


# ========== MAIN =============
f = sys.argv[1] # icsd_LEAFa.pickle
try:
    df = pd.read_csv(f)
except:
    df = pd.read_pickle(f)
print(df.head())

df = get_st(df)
X = np.asarray([np.array(i) for i in df['vectors']])
print(X.shape)
X = tsne(X,rs=0)
df['PCA_1'] = X[:,0]
df['PCA_2'] = X[:,1]
df.to_pickle('icsd_LEAFa_tsne_rs0.pickle')

fig = px.scatter(df, x='PCA_1', y='PCA_2', color='structure_type', symbol='crystal_system')
fig.update_layout(coloraxis_colorbar_x=-0.15)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
