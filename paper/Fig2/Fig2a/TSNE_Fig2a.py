# Figure 2a. sklearn v.1.5.1
import sys
import matplotlib.pyplot as plt
from pymatgen.core import Element #, ElementBase
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
print(sklearn.__version__)
atoms = open('atoms.txt').readlines()
atoms = [a.strip() for a in atoms]

def tsne(X, rs):
    X_embedded = TSNE(random_state=rs, n_components=2, learning_rate='auto',init='pca', perplexity=3).fit_transform(X) 
    #print(X_embedded.shape)
    return X_embedded

def prop(elements):
    props = [Element(e).Z for e in elements] 
   #props = [Element(e).atomic_radius_calculated for e in elements] 
   # props = [Element(e).average_ionic_radius for e in elements] 
    groups = [Element(e).group for e in elements] 
    rows = [Element(e).row for e in elements] 
    return props, groups, rows

def chemistry(elements):
    chems = []
    for e in elements:
        #if   Element(e).is_alkali: chems.append('alkali')
        #elif Element(e).is_alkaline: chems.append('alkaline')
        #if Element(e).is_chalcogen and e not in ['S', 'O']: chems.append('chalcogen')
        if Element(e).is_halogen: chems.append('halogen')
        elif Element(e).is_metal: chems.append('metal')
        #elif Element(e).is_lanthanoid: chems.append('lanthanoid')
        elif Element(e).is_noble_gas: chems.append('noble_gas')
        #elif Element(e).is_post_transition_metal: chems.append('post_transition_metal')
        elif Element(e).is_metalloid: 
            chems.append('metalloid')
            print(e)
        elif e in ['H', 'C', 'N', 'P', 'O', 'S', 'Se', 'Te']:
            chems.append('nonmetals and chalcogens')

    return chems

f = 'LEAF_average.csv' #sys.argv[1]

df = pd.read_csv(f)
df = df[df['element'].isin(atoms)]
#print(df.head())

atoms = df['element']
X = df.iloc[:,1:].to_numpy()

props, groups, rows = prop(atoms)
chem = chemistry(atoms)
symbols = ['circle', 'square', 'diamond', 'cross', 'x','triangle-up', 'pentagon'] #, hexagram, star]

for rs in [14]: #range(1): # 13
    X = tsne(X, rs)
    df = pd.DataFrame({'element':atoms, 'x':X[:,0], 'y': X[:, 1], 'property':props, 'group': groups, 'row': rows, 'chemistry': chem })
#    df = df.sort_values(by=['chemistry'])
    print(df.head())
    marker_symbol = {s: chem for s, chem in zip(symbols, list(set(df.chemistry)))}
    print(marker_symbol)

    color_scheme = px.colors.qualitative.Plotly[:18]  # Using a color scheme from Plotly
    fig = px.scatter(df, x='x', y='y', size=props, color='group', symbol = 'chemistry', color_continuous_scale='turbo')
#    fig = px.scatter(df, x='x', y='y', size=props, color='group', symbol = 'chemistry', symbol_map=marker_symbol)
    fig.update_layout(coloraxis_colorbar_x=-0.15)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.update_layout(
    title=False,
    xaxis=dict(
        showticklabels=False,
        showline=False,
        zeroline=False,
        showgrid=False
    ),
    yaxis=dict(
        showticklabels=False,
        showline=False,
        zeroline=False,
        showgrid=False
    )
    )
#    fig.update_traces(marker_symbol = symbols)
    fig.show()

#plt.scatter(X[:,0], X[:,1], c)
#plt.show()
