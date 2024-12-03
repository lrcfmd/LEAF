import pandas as pd
import sys
import numpy as np
from featurize_compositions import *
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc
from scipy.stats import wasserstein_distance as wass
from ElMD import ElMD
from sklearn.metrics.pairwise import cosine_similarity as cos
#from pdd import emd

df = pd.read_pickle('../SMACT_PREDICTED_DB.pickle')
true = df['structure_type'].to_list()
phase = df['formula_pretty'].to_list()

prediction = []
predicted_structure = []

#phase = matrix(phase)
phase = onehot(phase, atomfile='leaf+.csv' )

for i,f in enumerate(phase):
    test = phase[:i] + phase[i+1:]

    # EMD - minimizing score
    #scores = [emd(f, t) for t in test]
    #best = np.argmin(np.array(scores))

    # Cosine similarity - maximizing score
    scores = [cos(f.reshape(1,-1),t.reshape(1,-1)) for t in test]
    best = np.argmax(np.array(scores))

    prediction.append(test[best])
    predicted_structure.append(true[best])

ac = acc(true, predicted_structure)
mc = mcc(true, predicted_structure)
print(ac, mc)
