#  Feed a list of compositons and a list of elemental features (LEAF and return featurized compositions

import sys
from pymatgen.core import Composition
import numpy as np
import pandas as pd

def li_cat_anion(df, atoms, envs):
    """ encoding of compositions where vectors for
    a) Li
    b) average of cations
    c) average of anions
    are concatenated.
    E.g. if envs = LEAF59,
    then vector of length = 59*3 is returned """

    anions = ['I', 'Cl', 'O', 'S', 'Br', 'F']

    feat_comps = []
    vectors = []

    for composition in df['composition']:
  
        try:
            comp = Composition(composition)
            elements = [el.name for el in comp]
        except:
            continue
  
        if not set(elements).issubset(set(atoms)): continue
  
        feat_comps.append(composition)
        # amount - stoichiometry in reduced formula
        #amount = [float(comp.to_reduced_dict[el]) for el in elements]
        # amount - stoichiometry in normalised formula
        amount = [float(comp.fractional_composition.to_reduced_dict[el]) for el in elements]
  
        for el, am in zip(elements, amount):
            if el == 'Li':
                arg = list(atoms).index('Li')
                li = env[arg,:] * am
            elif el in anions:
                arg = list(atoms).index(el)
                ans = env[arg,:] * am
            else:
                arg = list(atoms).index(el)
                cats = env[arg,:] * am

#        vectors.append(li)
#        vectors.append([i for i in np.concatenate([li, cats, ans])])
        vectors.append([i for i in np.concatenate([li, cats])])
#        vectors.append([i for i in np.concatenate([li, ans])])

    return feat_comps, vectors


def phasehot(df, atoms, envs):
    """ create onehot encoding for compostions to later:
    by matmul: compositions @ atom2vec
    arg compsitions: list of compositions
    arg atoms: list of all supported atoms
    return: numpy array with stoichiometries at corresponding indexes
    """

    print("One-hot encoding of the phase fields ...")

    compositions = df['composition']
    onehot = np.zeros((len(compositions), len(atoms)))
    feat_comps = []

    for i, composition in enumerate(compositions):
         try:
             comp = Composition(composition)
             elements = [el.name for el in comp]
         except:
             continue

         if not set(elements).issubset(set(atoms)): continue

         feat_comps.append(composition)
         indexes = np.array([list(atoms).index(el) for el in elements])
         amount = [float(comp.to_reduced_dict[el]) for el in elements]
         onehot[i, indexes] = amount
    onehot = onehot[~np.all(onehot == 0, axis=1)]
    result =  onehot @ envs
    return result, feat_comps 


if __name__=='__main__':
    try:
        atomfile = sys.argv[1]
    except:
        print('No atom features provided. Reading from LEAF59.csv')
        atomfile = 'leaf+.csv'
        #atomfile = 'LEAF59.csv'
        #atomfile = 'LEAF_NEIGH.csv'
    try:
        comfile = sys.argv[2]
    except:
        print('No compositions provided. Reading from ../LiIon_classes_roomT.csv')
        comfile = '../LiIon_classes_roomT.csv'
   
    # list of chemical elements
    atoms = open('atoms.txt').readlines()
    atoms = [a.strip() for a in atoms]

    # list of atomic features
    print(f'Reading features from {atomfile}')
    df = pd.read_csv(atomfile)
    df = df[df['element'].isin(atoms)]
    atoms = df['element']
    print(df.head())
    print('Atoms considered:', atoms)
    env = df.iloc[:,1:].to_numpy()
    print('N features:', env.shape)


    # list of compositions
    comps = pd.read_csv(comfile)
    #X, feat_comps = phasehot(comps, atoms, env)
    feat_comps, X = li_cat_anion(comps, atoms, env)

    # compile results
    print('Original N compositions:', comps.shape)
    comps = comps[comps['composition'].isin(feat_comps)]
    print('Featurized N compositions:', comps.shape)
    comps['vectors'] = [i for i in X]
    print(comps.head())
    comps.to_pickle('Featurized_compositions_leaf+_concat_Li_cations_fract.pickle')
    #comps.to_pickle('Featurized_compositions_leaf+_concat_Li_cations.pickle')
    #comps.to_pickle('Featurized_compositions_leaf+_concat_Li_anions.pickle')
    #comps.to_pickle('Featurized_compositions_leaf+_concat_Li_cations_anions.pickle')

    #comps.to_pickle('Featurized_compositions_LEAFsa.pickle')
    #comps.to_pickle('Featurized_compositions_leaf+_concat_Li_cat.pickle')
    #comps.to_pickle('Featurized_compositions_magpie_concat_Li_cat.pickle')
    #comps.to_pickle('Featurized_compositions_mag2vec.pickle')
    #comps.to_pickle('Featurized_compositions_leaf_neigh_magpie.pickle')
