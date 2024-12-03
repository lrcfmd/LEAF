import sys
import pandas as pd
import numpy as np
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

featurizer = OPSiteFingerprint() 
lostops = featurizer.feature_labels()


# read lostops, conductivity and feature importance files
fi = pd.read_csv('calculated_feature_importance.csv')
#fi = fi.sort_values('importance', ascending=True)
print(fi.head())
cf = pd.read_csv('LiION_roomT.csv')
lf = pd.read_pickle('lostops_stats.pickle')
lf = lf.drop([779])


# averege features over instances of Li in compounds
lf['Li_features'] /= lf['natoms_insum']
#print(lf.head())

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

#lf['Li_features'] = lf['Li_features'].apply(lambda x: x[fi['feature'].to_numpy()])
lf['top_features'] = lf['Li_features'].apply(lambda x: np.argmax(x))
lf['sum_features'] = lf['Li_features'].apply(lambda x: sum(x[np.argsort(x)[:10]]))
lf['best_feature'] = lf['Li_features'].apply(lambda x: max(x)/sum(x))

lf = lf.dropna()

#print(len(lf['sum_features']), len(lf['top_features']), len(lf['target']))

# get linear regression line model:
LR = LinearRegression()

X = lf['top_features'].to_numpy()[:, np.newaxis]
LR.fit(X, lf['target'].values)
#plt.plot(X, LR.predict(X), '-g')


# Plot Scatter
cmap = plt.get_cmap('jet', max(lf['top_features']))

plt.scatter(lf['top_features'], lf['target'], s =1000*lf['best_feature'], c=lf['top_features'].to_numpy(), cmap=cmap)
#plt.scatter(lf['top_features'], lf['target'], c=lf['top_features'].to_numpy(), cmap=cmap)
#plt.scatter(lf['sum_features'], lf['target'], c=lf['top_features'].to_numpy(), cmap=cmap)
#plt.scatter(lf['best_feature'], lf['target'], s =100*lf['best_feature'], c=lf['top_features'].to_numpy(), cmap=cmap)
plt.xlabel('Li local structure environment', fontsize=14)
plt.ylabel(r'log($\sigma\;/\;S\;cm^{-1}$)', fontsize=14)
plt.colorbar()
plt.show()

sys.exit(0)
