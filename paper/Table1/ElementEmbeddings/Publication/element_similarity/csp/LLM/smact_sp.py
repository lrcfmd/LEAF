#!/usr/bin/env python

import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc
from monty.serialization import loadfn
from pymatgen.core.structure import Structure as pmg_structure
from pymatgen.core import Composition
import pandas as pd

### Load the data
print('Here we load the data from our radius ratio rules experiment.')
df = loadfn("../df_final_structure.json")
df["structure"] = df["structure"].apply(lambda x: pmg_structure.from_str(x, fmt="json"))
df['species'] = df["formula_pretty"].apply(lambda x: list(Composition(x).as_dict().keys()))
df['formula_pretty'] = df["formula_pretty"].apply(lambda x: Composition(x).formula)


### Load gnome probabilities
probs = pd.read_csv('Matscibert_subs_2.csv')

# Create a dictionary to map between material formula and ST
comp_to_st = {}
for i, row in df.iterrows():
    comp_to_st[row["formula_pretty"]] = row["structure_type"]

# Predict str type:
predicted = []
for i, row in df.iterrows():
    bf = df[~df['formula_pretty'].isin([row['formula_pretty']])]
    # get species intersection
    species = set(row['species'])
    keep_el = [species.intersection(set(spec)).pop() for spec in bf['species']
            if species.intersection(set(spec))]
    candidate_species = [
            [set(spec).difference(species).pop(), species.difference(set(spec)).pop()] 
            for spec in bf['species']
            if species.intersection(set(spec))
            ]
    candidate_species = [' '.join(sorted(l)) for l in candidate_species]
    sub_probs = [probs[probs['pairs'] == specs]['sub'].values[0]
            for specs in candidate_species]
    maxarg = np.argmax(sub_probs)
    best = np.array(candidate_species)[maxarg]
    sub_el = set(best.split(" ")).difference(species).pop()
    pred_comp = " ".join([np.array(keep_el)[maxarg], sub_el])
    print(row['formula_pretty'], Composition(pred_comp).formula)
    predicted.append(comp_to_st[Composition(pred_comp).formula])

df['predicted'] = predicted

### Metrics
true_label = df["structure_type"].values
pred_label = df["predicted"].values
mc = mcc(true_label, pred_label)
ac = acc(true_label, pred_label)
print('Accuracy:', round(ac*100, 2), 'MCC:', round(mc, 2))
sys.exit(0)
