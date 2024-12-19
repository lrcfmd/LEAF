# sklearn v.1.5.1
import sys
import matplotlib.pyplot as plt
from pymatgen.core import Composition

import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px


# ========== MAIN =============
f = 'icsd_LEAFa_tsne_rs0.pickle'
df = pd.read_pickle(f)
print(df.head())
print(df.shape)
X = np.asarray([np.array(i) for i in df['vectors']])
fig = px.scatter(df, x='PCA_1', y='PCA_2', color='structure_type', symbol='crystal_system')
fig.update_layout(coloraxis_colorbar_x=-0.15)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
