import sys
import pandas as pd
import numpy as np
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
#import plotly.express as plt
import matplotlib.pyplot as plt

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

# get structure finger prints
#fi['importance'] -= np.average(fi['importance'])

lf['sum_features'] = lf['Li_features'].apply(lambda x: sum(x[fi['feature'].to_numpy()] * fi['importance'].to_numpy()))
lf['top_features'] = lf['Li_features'].apply(lambda x: sum(x[fi['feature'].to_numpy()][:17] * fi['importance'].to_numpy()[:17]))
#print(lf.head())
plt.scatter(lf['sum_features'].to_numpy(), lf['target'], s=1500*lf['top_features'].to_numpy(), c='maroon')
#plt.scatter(lf['top_features'].to_numpy(), lf['target'])
plt.xlabel('Li environment fingerprint', fontsize=14)
plt.ylabel(r'log$_{10}({\sigma})$', fontsize=14)
plt.show()

sys.exit(0)

for b in best:
    x = [a[b] for a in lf['Li_features'].values]
    plt.scatter(x, lf['target'].values)
    plt.title(f'{lostops[b]}')
    plt.show()

#for i, l in enumerate(lostops):
#    print(i, l)


