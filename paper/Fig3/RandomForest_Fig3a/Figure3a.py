import sys
from matminer.featurizers.site.fingerprint import OPSiteFingerprint
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_pickle('features/Featurized_compositions_concat_Li_and_cat.pickle')

# Split the data into training and testing sets
X = data.drop('class>1e-4', axis=1)
X = np.asarray([np.asarray(x) for x in X['vectors']])
y = data['class>1e-4'].to_numpy()

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='accuracy')
f1_scores = cross_val_score(rf_classifier, X, y, cv=cv, scoring='f1')

print("Cross-Validation Accuracy Scores:")
print(scores, f1_scores)
print(f"Mean Accuracy: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")
print(f"Mean F1: {np.mean(f1_scores):.2f} (+/- {np.std(scores):.2f})")

rf_classifier.fit(X, y)
feature_importances = rf_classifier.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
num_features_to_display = 10

# Display or visualize the most important features
names = ['Li_' + f for f in OPSiteFingerprint().feature_labels()]
names += ['cat_' + f for f in OPSiteFingerprint().feature_labels()]

top_features = sorted_indices[:num_features_to_display]
top_feature_importances = feature_importances[top_features]
names_importances = np.array(names)[top_features]

print("\nTop {} Features:".format(num_features_to_display))
for feature, importance in zip(top_features, top_feature_importances):
    print(f"{feature}, {names[feature]}, {importance:.4f}")

sns.set(style="whitegrid")  # Set the plot style
plt.plot(np.arange(-2, len(names_importances)+2),  \
        0.013*np.ones(len(names_importances)+4), 'r--',zorder=-1)
        
sns.barplot(x=names_importances, y=top_feature_importances, palette="Blues_r")
# Add labels and title
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.xticks([])
plt.show()
