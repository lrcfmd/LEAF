import sys, os
import pandas as pd
import pymatgen
#from pymatgen.analysis.local_env import *
from pymatgen.io.cif import CifParser
#from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
#from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.site.fingerprint import OPSiteFingerprint, VoronoiFingerprint


featurizer = OPSiteFingerprint() 
f = featurizer.feature_labels()
print(f)
print(featurizer.gone)
sys.exit(0)


for p, d, f in os.walk('cifs/'):
    for file in f:
        print(file)
        parser = CifParser(f'cifs/{file}')
        try:
            structure = parser.get_structures()[0]
            features = []
            name = file.split('.cif')[0]
            print(name)
            for i in range(len(structure)):
                features.append(featurizer.featurize(structure, i))
            df = pd.DataFrame({'composition': name, 'structure': structure, 'features':features}) 
            df.to_pickle(f'features/{name}.pickle')
        except:
            continue

#mp = pd.DataFrame({'structure': structures})
#features = featurizer.featurize_dataframe(mp, col_id='structure',ignore_errors=True)
#mp['features'] =  features
#features = features.dropna()
#mp.to_csv('features_top100.csv', index=False)

sys.exit()

features = []
for i,structure in enumerate(structures):
    features = []
    for i in range(len(structure)): 
        features.append(featurizer.featurize(structure, i))
    df = pd.DataFrame({'structure': structure, 'features':features}) 
    df.to_pickle(f'str_features{i}.pickle')

sys.exit()

mp['features'] =  features
mp.to_csv('features_top100.csv', index=False)


features = featurizer.featurize_dataframe(mp, col_id='structure',ignore_errors=True)
features = features.dropna()

print(features.shape)
print(features.head())

#for i in features.iloc[1]: print(i)
