#  Feed a list of compositons and a list of elemental features (LEAF and return featurized compositions

import sys
from pymatgen.core import Composition
import numpy as np
import pandas as pd

def onehot(compositions, atomfile='leaf+.csv'):
    """ create onehot encoding for compostions to later:
    by matmul: compositions @ atom2vec
    arg compsitions: list of compositions
    return: numpy array with stoichiometries at corresponding indexes
    """
    # get {atom: vector} dict
    df = pd.read_csv(atomfile)
    df_transposed = df.transpose()
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    atom_dict = df_transposed.to_dict(orient='list')
    # form representation: (weigths*vectors)

    result = []
    for i, composition in enumerate(compositions):
         comp = Composition(composition)
         elements = [el.name for el in comp]
         comp_dict = comp.as_dict()
         vector = np.zeros(df_transposed.shape[0])
         for el in elements:
             vector += comp_dict[el] * np.array(atom_dict[el])

         result.append(vector)
    return result


def matrix(compositions, atomfile='leaf+.csv'):
    # get {atom: vector} dict
    df = pd.read_csv(atomfile)
    df_transposed = df.transpose()
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    atom_dict = df_transposed.to_dict(orient='list')
    # form representation: (weigths, (vectors))

    result = []
    for i, composition in enumerate(compositions):
         comp = Composition(composition)
         elements = [el.name for el in comp]
         comp_dict = comp.as_dict()

         matrix = [] 
         for el in elements:
             vector = [comp_dict[el]] + atom_dict[el]
             matrix.append(vector)

         result.append(np.array(matrix))
    #return np.stack(result)
    return result






if __name__=='__main__':
    onehot(['AlCl'])
