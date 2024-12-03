import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity


ELEMENTS = tuple(
    'H|He|'
    'Li|Be|B|C|N|O|F|Ne|'
    'Na|Mg|Al|Si|P|S|Cl|Ar|'
    'K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|'
    'Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|I|Xe|'
    'Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|At|Rn|'
    'Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr|Rf|Db|Sg|Bh|Hs|Mt|Ds|Rg'.split('|')
)
df = pd.read_csv('GNoME_element_subs.csv')

# Atom2Vec files
a2v = {}
indices = open('atoms_index.txt').readlines()
vectors = open('atoms_vec.txt').readlines()
for i, v in zip(indices, vectors):
    symbol = ELEMENTS[int(i)]
    print(symbol)
    a2v[symbol] = [float(j) for j in v.split(" ")]

bf = pd.DataFrame(a2v)
print(bf.head())
bf = bf.transpose()
print(bf.head())
bf = bf.reset_index()
print(bf.head())
bf.to_csv('atom2vec.csv', index=False)

sys.exit(0)
dirs = {}
similarities = []
for p in df['pairs']:
    a = p.split(" ")[0]
    b = p.split(" ")[1]
    sim = cosine_similarity([a2v[a], a2v[b]])
    print(a, b, sim[0][1])
    similarities.append(sim[0][1])
df['sub_a2v'] = similarities
df.to_csv('Atom2vec_subs.csv', index=False)
