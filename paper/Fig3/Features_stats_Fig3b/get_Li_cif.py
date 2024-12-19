import os
import re
import sys
import pandas as pd
import pymatgen
#from pymatgen.analysis.local_env import *
from pymatgen.io.cif import CifParser
#from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
#from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.site.fingerprint import OPSiteFingerprint, VoronoiFingerprint
from ICSDClient import *

def writeout(cifs, name, folder="./cifroom/"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if not isinstance(cifs, list):
        if cifs is None:
            print("Requires a valid cif string, this string is None. Ensure download was successful")
            return 
            
        cifs = [cifs]
    
    for i, cif in enumerate(cifs):
        filename = f"{name}_id{i}.cif"

        with open(os.path.join(folder, filename), "w") as f:
            for line in cif.splitlines():
                f.write(line + "\n")


client = ICSDClient(f"{your_API_key}", f"{your_ICSD_login_name}")

#from ase.data import chemical_symbols

#top = pd.read_csv('top100.csv')
top = pd.read_csv('LiION_roomT.csv')
tops = top['formula'].values
temp = top['Temperature (C)'].values

for s,t in zip(tops, temp):
    print(s, t)
    search = client.search(s)
    #search = client.advanced_search({'composition': s}, property_list=['Temperature'])
    print(search)
#    cifs = client.fetch_cifs(search)
    #writeout(cifs, s)
    #client.writeout(cifs)

client.logout()
sys.exit(0)

for cif in cifs[:1]:
    parser = CifParser(cif)
    structure = parser.get_structures()[0]
#    print(structure)


featurizer = OPSiteFingerprint() 
f = featurizer.feature_labels()


client.logout()
sys.exit(0)

#atoms = chemical_symbols[1:3]
#print(atoms)


# get data

# from cifs
parser = CifParser("icsd_083501.cif")
structure = parser.get_structures()[0]

# MP retrieval
#mp = MPDataRetrieval(api_key="4r0hw9Mm0wjGgkSeIE").get_dataframe(criteria='Ta-S', properties=['structure','formula'])
#print(mp.head())
#print(mp.shape)
#for com in mp.formula.values:
#    print([el for el in com.keys()])

# species = [str(s).split()[-1] for s in structure]
# species = [''.join([a for a in s if a.isalpha()]) for s in species]
# print(len(species))

#mp = pd.DataFrame({'structure': [structure]})

features = []
for i in range(len(structure)): features.append(featurizer.featurize(structure, i))
for i in features: print(i)
sys.exit()

features = featurizer.featurize_dataframe(mp, col_id='structure',ignore_errors=True)
features = features.dropna()

print(features.shape)
print(features.head())

#for i in features.iloc[1]: print(i)
