import sys
import numpy as np
from pymatgen.core import Composition
from smact import Element
from smact.structure_prediction import (
#    prediction,
    mutation,
#    structure,
)

class SubLEAF:
    """ Class predicts substitution probability between two lists of atoms
    based on the local environment-iduced atomic features (LEAFs).

    The work is presented in
    A. Vasylenko et.al 'Learning atoms from crystal structure' (2024)
    """

    def __init__(self, parent):
        self.sub = mutation.CationMutator.from_json(f"cosine_similarity_leaf/leaf+.json")
        self.parent = self.oxidize(parent)

    def oxidize(self, composition):
        dic = Composition(composition).oxi_state_guesses()
        if len(dic):
            dic = dic[0]
            signs = ['+' if ox > 0 else '-' for ox in dic.values()]
            oxis = [str(int(abs(i))) for i in dic.values()]
            symbols = list(dic.keys())
            elements = [e + o + s for e, o, s in zip(symbols, oxis, signs)]
            return elements
        else:
            print('Composition must be charge ballanced. Exiting')
            sys.exit(0) 

    def probability(self, child):
        child = self.oxidize(child)
        alt_specs = list(set(self.parent) - set(child))
        probs = [self.sub.cond_sub_probs(a).loc[a] for a in alt_specs]
        print (child, self.parent, probs)
        return np.prod(probs)
  

if __name__=="__main__":
    a = SubLEAF('LiNiPO4')
    b = a.probability('LiFePO4')
    print(b)
    z = SubLEAF('CaTiO3')
    zz = z.probability('BaTiO3')
    print(zz)
