import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

datafile = sys.argv[1]  # Leaf_av_magpie
data = pd.read_pickle(datafile)

features = pd.read_csv('../features/leaf_av_magpie.csv')
features = features.columns[1:]

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
num_features_to_display = 12

# Display or visualize the most important features
top_features = features[sorted_indices[:num_features_to_display]]
top_feature_importances = feature_importances[sorted_indices[:num_features_to_display]]
top_features_i = sorted_indices[:num_features_to_display]

print(top_features)

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
