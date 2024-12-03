import sys
import pandas as pd
import numpy as np
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
#import plotly.express as plt
import matplotlib.pyplot as plt
import seaborn as sns

featurizer = OPSiteFingerprint() 
lostops = featurizer.feature_labels()


# read lostops, conductivity and feature importance files
fi = pd.read_csv('calculated_feature_importance.csv')
cf = pd.read_csv('LiION_roomT.csv')
lf = pd.read_pickle('lostops_stats.pickle')
lf = lf.drop([779])

# averege features over instances of Li in compounds
lf['Li_features'] /= lf['natoms_insum']

# add conductivity target to features data
target = []
for f in lf['formula']:
    af = cf[cf['formula']==f]['target'].values[0]
    target.append(af)
lf['target'] = target
lf = lf.sort_values(by='target', ascending=False)

# remove duplicates leaving most conductive examples
lf = lf.drop_duplicates(['formula'])

# reshuffle features in accord to importance
lf['top_features'] = lf['Li_features'].apply(lambda x: x[fi['feature'].to_numpy()][:9])
#lf['top_features'] = lf['Li_features'].apply(lambda x: x[fi['feature'].to_numpy()])
top_features = np.array([f for f in lf['top_features']]).T

colors = [[int(i) for i in lf['target']] for x in np.arange(1,10)]
#colors = [[int(i) for i in lf['target']] for x in np.arange(1,38)]
colors = np.array([color for row in colors for color in row])

plt.scatter([x*np.ones(34) for x in np.arange(1,10)], [i for i in top_features], c=colors, cmap='bwr')
#plt.scatter([x*np.ones(34) for x in np.arange(1,38)], [i for i in top_features], s=[100/abs(i) for i in colors], c=colors, cmap='bwr')
plt.colorbar()
plt.savefig('Fig3b_alt.png', dpi=400) 
#[f[:10]lf['top_features'].iloc[:10,:], c='maroon')
#plt.scatter(lf['top_features'].to_numpy(), lf['target'])
#plt.xlabel('Li environment fingerprint', fontsize=14)
#plt.ylabel(r'log$_{10}({\sigma})$', fontsize=14)
plt.show()
