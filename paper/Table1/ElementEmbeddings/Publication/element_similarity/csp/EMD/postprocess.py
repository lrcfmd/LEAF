import pandas as pd
import numpy as np
from ElMD import ElMD
#from sklearn.metrics.pairwise import accuracy, matthews_corr_coef
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import accuracy_score as acc

#df = pd.read_pickle('SMACT_PREDICTED_DB.pickle')
df = pd.read_pickle('SMACT_PREDICTED_EMDleaf.pickle')
print(df.columns)
print(df.head())
true = df['structure_type'].to_list()
predicted_s = df['EMD_structure_type'].to_list()
predicted_s = df['leaf_neighbour_struct'].to_list()

ac = acc(true, predicted_s)
print(ac, mcc(true, predicted_s))
