import torch
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
import sys
import numpy as np
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc
from monty.serialization import loadfn
from pymatgen.core.structure import Structure as pmg_structure
from pymatgen.core import Composition
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

def tokenize(pair):
    element, sub = tuple(pair.split())
    #tokens = tokenizer(f'Estimate a probability for the chemical element {element} to form a stable compound with {sub}')
    tokens = tokenizer(f'What is a likely crystal structure of a compound where the chemical element {element} can be substituted with {sub}')
    t1 = tokens[0][0]
    #tokens = tokenizer(f'Estimate a probability for the chemical element {sub} to form a stable compound with {element}')
    tokens = tokenizer(f'What is a likely crystal structure of a compound where the chemical element {sub} can be substituted with {element}')
    t2 = tokens[0][0]
    cos = cosine_similarity([t1, t2])[0][1]
    return cos

df = pd.read_csv('../gnome/GNoME_element_subs.csv')

# ==============LOAD LLM==================
tokenizer = pipeline("feature-extraction", model="m3rg-iitd/matscibert")
features = []
for pair in df['pairs']: 
    features.append(tokenize(pair))

bf = pd.DataFrame({'pairs': df.pairs.to_list(), 'sub': features})
bf.to_csv('Matscibert_subs_2.csv', index=False)
