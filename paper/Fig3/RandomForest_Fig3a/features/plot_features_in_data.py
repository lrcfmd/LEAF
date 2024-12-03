import seaborn as sns
import matplotlib.pyplot as plt
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
import sys
import pandas as pd
import numpy as np

features = {'15': 0.0751, '22': 0.06129999999999999, '10': 0.0563, '4': 0.0494, 'Z': 0.0459, '36': 0.0441}

try:
    df = pd.read_csv(sys.argv[1])
    features = [i for i in features if i.isdigit()]
    data = [df[i].to_numpy() for i in features]
except:
    df = pd.read_pickle(sys.argv[1])
    df = df['vectors'].T.to_numpy()
    features = [int(i) for i in features if i.isdigit()]
    data = [df[i] for i in features] 

names = [i for i in OPSiteFingerprint().feature_labels()]
names = [names[i] for i in features]

# Create a Seaborn bar plot
sns.set(style="whitegrid")  # Set the plot style

# You can customize the color palette using sns.color_palette()
# Example: sns.color_palette("Blues_d")

# Create the bar plot
for i in range(len(data)):
    sns.histplot(data[i], bins=10, kde=True, color="skyblue")
    # Add labels and title
    plt.xlabel(names[i]) #"Square pyramid CN5 value")
    plt.ylabel("Number of compositions")

# Show the plot
    plt.show()
