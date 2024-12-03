import sys, os
import numpy as np
import pandas as pd
import pymatgen
#from pymatgen.analysis.local_env import *
from pymatgen.io.cif import CifParser
#from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
#from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.site.fingerprint import OPSiteFingerprint, VoronoiFingerprint


featurizer = OPSiteFingerprint() 
names = featurizer.feature_labels()
f_all = []
compositions = []
natoms = []

for p, d, f in os.walk('features/'):
    for file in f:
        formula = file.split('_')[0]
        print(formula)
        features = np.zeros(len(names))
        df = pd.read_pickle(f'features/{file}')
        n_li = 0
        for i, row in df.iterrows():
            atom = row['structure'].species.formula
            if 'Li' in atom:
                #print(atom)
                features += np.array(row['features'])
                n_li += 1
        
        natoms.append(n_li)
        compositions.append(formula)
        f_all.append(features)

df = pd.DataFrame({'formula': compositions, 'Li_features': f_all, 'natoms_insum': natoms})
df.to_csv('lostops_stats.csv', index=False)
df.to_pickle('lostops_stats.pickle')
