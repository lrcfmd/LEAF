import sys
import numpy as np
import pandas as pd
from pymatgen.core import Composition as C
from pymatgen.core import Element
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef as mcc
import matplotlib.pyplot as plt


data = pd.read_pickle('features/Featurized_compositions_concat_Li_only.pickle')
print(data.columns)
#print(data)


# Split the data into training and testing sets
X = data.drop('class>1e-4', axis=1)
X = np.asarray([np.asarray(x) for x in X['vectors']])
#X = np.asarray([np.asarray(x) for x in X['at_fraction']])
#X = np.asarray([np.asarray(x) for x in X['wt_fraction']])

# Random
#X = np.asarray([np.random.rand(37) for c in data['composition']])
print('VECTORS example, entry 0:', X[0], X.shape)

#if a single feature: reshape:
if X.shape[1] == 1:
    X = X.reshape(-1, 1)
y = data['class>1e-4'].to_numpy()


more = y[y>0]
less = y[y==0] 
print('class>1e-4:' ,len(more), 'less:', len(less))
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='accuracy')
mcc_scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='matthews_corrcoef')
f1_scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='f1')

print("Cross-Validation Accuracy Scores:")
print(scores, f1_scores, mcc_scores)
print(f"Mean Accuracy: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
print(f"Mean MCC: {np.mean(mcc_scores):.2f} (+/- {np.std(scores):.2f})")
#print(f"Mean F1: {np.mean(f1_scores):.2f} (+/- {np.std(scores):.2f})")


sys.exit(0)
# Optional: Visualize feature importances for a single Random Forest
rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
num_features_to_display = 10

# Display or visualize the most important features
top_features = X.columns[sorted_indices[:num_features_to_display]]
top_feature_importances = feature_importances[sorted_indices[:num_features_to_display]]

print("\nTop {} Features:".format(num_features_to_display))
for feature, importance in zip(top_features, top_feature_importances):
    print(f"{feature}: {importance:.4f}")

# Optional: Visualize feature importances
plt.figure(figsize=(10, 6))
plt.title("Top Feature Importances")
plt.bar(range(num_features_to_display), top_feature_importances, align="center")
plt.xticks(range(num_features_to_display), top_features, rotation=45)
plt.tight_layout()
plt.show()
