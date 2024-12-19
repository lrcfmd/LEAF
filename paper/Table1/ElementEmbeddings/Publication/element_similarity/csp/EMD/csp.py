import pandas as pd
import numpy as np
from ElMD import ElMD
#from sklearn.metrics.pairwise import accuracy, matthews_corr_coef
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc

df = pd.read_pickle('../SMACT_PREDICTED_DB.pickle')
true = df['structure_type'].to_list()
phase = df['formula_pretty'].to_list()
check = df[['structure_type', 'formula_pretty']]
#print(check.head())

prediction = []
predicted_structure = []

for i,f in enumerate(phase):
    test = phase[:i] + phase[i+1:]
    x = ElMD(formula=f)
    scores = [x.elmd(t) for t in test]
    best = np.argmin(np.array(scores))
    prediction.append(test[best])
    predicted_structure.append(true[best])

check['EMDleaf_formula'] = prediction 
check['EMDleaf_structure'] = predicted_structure 

ac = acc(true, predicted_structure)
mc = mcc(true, predicted_structure)
print(ac, mc)
