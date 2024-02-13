import sys
import pandas as pd
import numpy as np
from pymatgen.core import Composition
from smact.structure_prediction import mutation

class LEAF:
    """ Class predicts substitution probability between two lists of atoms
    based on the local environment-iduced atomic features (LEAFs).

    The work is presented in
    A. Vasylenko et.al 'Learning atoms from crystal structure' (2024)
    """

    def __init__(self, parent, oxidations=False, data='../data//leaf+_dic.cvs'):
        self.leaf = pd.read_csv(data)
        self.sub = mutation.CationMutator.from_json(f"/Users/andrij/Programmes/LEAF/LEAF/Compositional_structural_metrics/cosine_similarity_leaf/leaf+.json")
        self.oxidations = oxidations

        self.representation = self.represent(parent)
        self.parent = self.oxidize(parent, self.oxidations)

    def oxidize(self, composition, oxidations=False):
        if not oxidations: 
            dic = Composition(composition).oxi_state_guesses()
            dic = dic[0]
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

    def represent(self, composition, kind='sum'):
        if kind == 'sum':
            element_amounts = Composition(composition).to_reduced_dict
            vector = [self.leaf[element].to_numpy() * amount for element, amount in element_amounts.items()]
            vector = np.stack(vector).sum(axis=0)
        #elif kind == 'concat': TODO
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
        print('Best probability:', best)
        return candidates, best


if __name__=="__main__":
    z = LEAF('CaTiO3')
    zz = z.sub_probability('SrTiO3') #oxidations={'Sr':2, 'Ti':3, 'O':-2})
    print(zz)
    
    #best_probs, best = z.best_substitutions(['Ca2+','Ti4+','O2-'],top=10)
    #best_probs, best = z.best_substitutions(['Ca2+','Ti4+'],top=10)
    z.best_substitutions(['Ca2+','Ti4+'],top=10)

    print(z.representation)

    #df = z.sub.lambda_tab
    #print(df)
