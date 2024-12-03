""" Local environment-induced atomic features
"""
import os
import sys
import numpy as np
import pandas as pd
from ase.data import chemical_symbols
from pymatgen.core import Composition
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import *
from matminer.featurizers.site.fingerprint import OPSiteFingerprint, VoronoiFingerprint
from smact.structure_prediction import mutation

class Leaf:
    """ Class predicts substitution probability between two lists of atoms
    based on the local environment-iduced atomic features (LEAFs).

    The work is presented in
    A. Vasylenko et.al 'Learning atoms from crystal structure' (2024)
    """

    def __init__(self, parent, oxidations=False, data='data/leaf+_dic.csv'):
        if isinstance(data, str):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_path = os.path.join(base_dir, data)
            if 'csv' in data:
                self.leaf = pd.read_csv(data_path)
            elif 'pickle' or 'pkl' in data:
                self.leaf = pd.read_pickle(data_path)
            json_data = os.path.join(base_dir, "data/leaf+.json")
            self.sub = mutation.CationMutator.from_json(json_data)

        elif isinstance(data, pd.DataFrame):
            self.leaf = data

        self.oxidations = oxidations
        self.formula = parent

        self.representation = self.represent(parent)
        self.parent = self.oxidize(parent, self.oxidations)

    @staticmethod
    def calculate_lostops_from_cif(cif):
        """ Use matminer's implementation for calculating LOStOPs for a given structure"""
        from matminer.featurizers.site.fingerprint import OPSiteFingerprint
        from pymatgen.io.cif import CifParser
       
        featurizer = OPSiteFingerprint()
        structure = CifParser(f'{cif}').get_structures()[0]

        print(f'Calculating LOStOPs for {cif}:')
        print(featurizer.feature_labels())
        features = []
        for atomsite in structure:
            features.append(featurizer.featurize(structure, atomsite))
        return features

    def oxidize(self, composition, oxidations=False):
        if not oxidations: 
            dic = Composition(composition).oxi_state_guesses()
            if len(dic):
                dic = dic[0]
            else:    
                print(composition, ' does not seem to be balanced; no charges were provided.')
        else:
            dic = oxidations
        if len(dic):
            signs = ['+' if ox > 0 else '-' for ox in dic.values()]
            oxis = [str(int(abs(i))) for i in dic.values()]
            symbols = list(dic.keys())
            elements = [e + o + s for e, o, s in zip(symbols, oxis, signs)]
            return elements
        else:
            print('Composition must be charge ballanced. Exiting')
            sys.exit(0) 

    def sub_probability(self, child, oxidations=False, partial=False):
        child = self.oxidize(child, oxidations=oxidations)
        best_probs = [self.sub.cond_sub_probs(a).loc[a] for a in self.parent] 
        probs = [np.round(self.sub.cond_sub_prob(a, b), 6) for a, b in zip(self.parent, child)]
        print (child, self.parent, probs)
        return probs if partial else np.prod(probs) 

    def represent(self, composition):
        element_amounts = Composition(composition).to_reduced_dict
        vector = [self.leaf[element].to_numpy() * amount for element, amount in element_amounts.items()]
        vector = np.stack(vector).sum(axis=0)
        return vector

    def similarity(self, child):
        parent = self.representation
        child = self.represent(child)
        #return np.round(np.exp(np.dot(parent, child) / np.linalg.norm(parent) / np.linalg.norm(child)), 6)
        return np.round(np.power(10, (np.dot(parent, child) / np.linalg.norm(parent) / np.linalg.norm(child))), 6)

    def best_substitutions(self, elements=None, top=1):
        """ Calculates top = x best substitutions for each element in a composition """
        if elements is None:
            elements = self.parent

        candidates = []
        for el in elements:
            prob_i = self.sub.cond_sub_probs(el).to_numpy()
            args = np.argsort(prob_i)[::-1][:top*5]
            probs = prob_i[args]
            els = self.sub.lambda_tab.columns[args]
            considered = []
            cand_i = []
            for eli, p_i in zip(els, probs):
                name = ''.join([a for a in eli if a.isalpha()])
                if name in considered: continue
                considered.append(name)
                cand_i.append((name, round(p_i,6)))
                if len(cand_i) == top: break
            candidates.append(cand_i)

        for el, cand in zip(elements, candidates):
            print(el, cand)

        best = []
        for c in candidates:
            _, p = c[0]
            best.append(p)
        best = np.prod(np.array(best))
        print(f'Best probability of {len(elements)} elemental subs:', best)
        return candidates, best

    def cosine_similarity(self, matrix):
        v = self.representation
        return  np.dot(matrix, v) / np.linalg.norm(v) / np.linalg.norm(matrix, axis=1)

    def leafy_icsd(self, file='data/icsd_compositions.csv'):
        """ Represent all compositions in ICSD with LEAFs """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(base_dir, file)
        icsd = pd.read_csv(file)
        icsd['leaf'] = icsd['composition'].apply(lambda x: self.represent(x))
        newfile = 'data/icsd_unique_struct_leaf+.pickle'
        icsd.to_pickle(f'{base_dir}/{newfile}')
        return icsd

    def best_icsd_similarity(self, top=1, drop_self_duplicates=True, unique_structure_type=False):
        """ Calculates similarity of top=x best compositions in ICSD """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data = 'data/icsd_unique_struct_leaf+.pickle'
        datapath = os.path.join(base_dir, data)
        if os.path.isfile(datapath):
            icsd = pd.read_pickle(datapath)
        else:
            icsd = self.leafy_icsd()
        icsd_leaf = np.vstack(icsd['leaf'].values)
        icsd['similarities'] = self.cosine_similarity(icsd_leaf)
        icsd = icsd.sort_values(by=['similarities'], ascending=False)

        if drop_self_duplicates:
            icsd = icsd[~icsd['composition'].isin([self.formula])]
        if unique_structure_type:
            icsd = icsd.drop_duplicates(subset=['structure_type'])

        bf = icsd.head(top)
        print(f'Top {top} most similar reported compositions:') 
        print('%20s %10s %25s %20s' %('composition', 'cosine similarity','structure type','crystal system'))
        for i, row in bf.iterrows():
            print('%20s %10f %30s %20s' 
                  %(row['composition'], round(row['similarities'],6), row['structure_type'], row['crystal_system']))

#-------------------------------------------------------------------------
class CreateLeaf:
    """
    Creates a matrix of one-hot encoded atomic representation
    or a dictionary of averaged LEAFs

    Values: lists of local environment features (lostops, voronoi tes.)
    Keys:   atomic elements
    """

    def __init__(self, featurizer=OPSiteFingerprint(), icsdfile=None, onehot=None):
        self.featurizer = featurizer
        self.features_names = self.get_features_names()
        if icsdfile:
            self.icsd = pd.read_pickle(icsdfile)
        if onehot:
            self.onehot = pd.read_pickle(onehot)
        else:
            print('Computing LEAFs from CIFs \
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
            \nInitiating an empty dictionary  {atom: {} for atom in chemical_symbols}')
            self.onehot = {atom: {} for atom in chemical_symbols}

    def get_features_names(self):
        features = [i for i in self.featurizer.feature_labels()]
        return features

    @staticmethod
    def get_species(site):
        species = str(site.species).split()
        return [''.join([a for a in s if a.isalpha()]) for s in species \
                if ''.join([a for a in s if a.isalpha()]) in chemical_symbols]

    @staticmethod
    def readfile(list_cifs, mode='dir'):
        """'list':  read file with list of cifs e.g. 1.dat
            'dir':  read cifs from a directory  """

        if mode == 'list':
            cifs = open(list_cifs, 'r').readlines()
            cifs = [i.strip() for i in cifs]
            order = list_cifs.split('.')[0]
        elif mode == 'dir':
            cifs = []
            for p, d, fi in os.walk('../{list_cifs}'):
                for f in fi:
                    print(f)
                    cifs.append(f'../{list_cifs}/{f}')
            order = len(cifs)
        else:
            raise RuntimeError("Only 'list' or 'dir' mode are recognised when reading CIFs")
        return cifs, order

    @staticmethod
    def select_features(i, s):
        """
        from all voronoi features select:
        'Voro_vol_sum', 'Voro_area_sum', 'Voro_vol_mean', 'Voro_vol_minimum',
        'Voro_vol_maximum', 'Voro_area_mean', 'Voro_area_minimum', 'Voro_area_maximum',
        'Voro_dist_mean', 'Voro_dist_minimum', 'Voro_dist_maximum'
        """
        l = list(VoronoiFingerprint().featurize(s, i))
        b = l[16:19] + l[20:23] + l[24:27] + l[28:31]
        return np.array(b)

    def generate_from_cifs(self, list_cifs, mode='dir'):
        """ process CIFs
        from a list: mode='list'
        from a directory: mode='dir'

        average features over a number of occurences of the elements in CIFs
        write results into a dictionary """
        base_ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        list_cifs = os.path.join(base_, list_cifs)
        if not os.path.exists(list_cifs):
            raise ValueError(f'{list_cifs} does not exist')
        
        icsd = {'composition':[],'structure':[]}

        leaf = {atom: np.zeros(37) for atom in chemical_symbols}
        # dictionary of number of occurence
        nleaf = {atom: 0 for atom in chemical_symbols}
       
        if mode == 'list':
            cifs, order = self.readfile(list_cifs, mode=mode)
        elif mode == 'dir':
            print(f'Reading CIFs from {list_cifs}')
            cifs = []
            for _, _, fi in os.walk(f'{list_cifs}'):
                for f in fi:
                    print(f)
                    cifs.append(f'{list_cifs}/{f}')
            order = len(cifs)
        else:
            raise ValueError('generate_from_cifs: wrong value for "mode", ("dir", "list") are allowed')

        for cif in cifs:
            try:
                structure = CifParser(cif).get_structures()[0]
            except Exception:
                continue
            icsd['composition'].append(str(structure.composition))
            icsd['structure'].append(structure) 
            features = []
            for i, s in enumerate(structure):
                species = self.get_species(s)
                for element in species:
                    leaf[element] += self.featurizer.featurize(structure, i)
                    nleaf[element] += 1

        # average over instances
        for element in leaf:
            if nleaf[element]:
                leaf[element] /= nleaf[element]

        df = pd.DataFrame(leaf)
        ndf = df.loc[:, (df != 0).any(axis=0)]
        print(f'Computed LEAFs for {ndf.columns} elements') 
        print(f'Saving to LEAFs_{order}.pickle') 
        df.to_pickle(f'LEAFs_{order}.pickle')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        return df

    def get_structures(self, list_cifs=None):
        structures = []
        if list_cifs:
            cifs, order = self.readfile(list_cifs)
            for cif in cifs:
                try:
                    structures.append(CifParser(cif).get_structures()[0])
                except:
                    continue
        else:
            structures = self.icsd['structure'].values
        return structures

    def get_features(self, decimals=1, list_cifs=None):
        features = []
        elements = []
        structures = self.get_structures(list_cifs)

        for structure in structures:
            for i, s in enumerate(structure):
                species = self.get_species(s)
                for element in species:
                    features = [round(f,decimals) for f in self.featurizer.featurize(structure, i)]
                    self.expand_onehot(element, features)

    def expand_onehot(self, element, features):
        """ For all features 'feature_name' with values 'v'
        create one-hot columns 'feature_name_v'
        fill the number of occurences into self.onehot
        """
        for name, v in zip(self.features_names, features):
            if v:
                feature = name + '_' + str(v)
                if feature in self.onehot[element]:
                    self.onehot[element][feature] += 1
                else:
                    for atom in chemical_symbols:
                        self.onehot[atom][feature] = 0

    def sort_features(self):
         """ sort by features names
         return dictionary[atoms] = 'features values' """
         for atom in self.onehot:
             self.onehot[atom] = {k: v for k,v in sorted(self.onehot[atom].items(), key=lambda x: x[0])}

         return {a: np.array(self.onehot[a].values()) for a in self.onehot}

    def save_onehot(self, order):
        onehot = pd.DataFrame(self.onehot).reset_index()
        onehot.to_pickle(f'onehot_decimals3_{order}.pickle')

if __name__=="__main__":
    # To create new features from the CIF files
    fname = sys.argv[1] # provide a list or directory with CIF files e.g. ../testcifs 
    newleaf = CreateLeaf()
    leafs_test = newleaf.generate_from_cifs(fname, mode='dir')

    # To get one-hot matrix leafs from the collected structures (icsd files):
    #newleaf = CreateLeaf(onehot=None, icsdfile=fname)
    #newleaf.get_features(decimals=2)
    #newleaf.save_onehot('new')

    # To employ precomputed LEAFs for applications

    z = Leaf('CaTiO3')

    zz = z.sub_probability('SrTiO3') #oxidations={'Sr':2, 'Ti':3, 'O':-2})
    print(zz)
    
    #best_probs, best = z.best_substitutions(['Ca2+','Ti4+','O2-'],top=10)
    best_probs, best = z.best_substitutions(['Ca2+','Ti4+'],top=10)

    print(z.representation)

    z.best_icsd_similarity(top=10, drop_self_duplicates=True, unique_structure_type=True)

    #df = z.sub.lambda_tab
    #print(df)
