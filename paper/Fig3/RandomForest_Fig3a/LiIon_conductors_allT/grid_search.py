import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


datafile = sys.argv[1]
data = pd.read_pickle(datafile)

# Split the data into training and testing sets
X = data.drop('class>1e-4', axis=1)
X = np.asarray([np.asarray(x) for x in X['vectors']])
y = data['class>1e-4'].to_numpy()

# Define the parameter grid for hyperparameter optimization
param_grid = {
    'n_estimators': [50, 100, 200],  # Different numbers of trees
    'max_depth': [None, 10, 20],     # Maximum depth of each tree
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],   # Minimum samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
    'random_state': [42]
}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Create the GridSearchCV object
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

# Perform hyperparameter optimization
grid_search.fit(X, y)

# Print the best hyperparameters and corresponding accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print("Best Hyperparameters:")
print(best_params)
print(f"Best Mean Accuracy: {best_accuracy:.2f}")
sys.exit(0)
# Optional: Visualize feature importances for the best model
best_rf_classifier = grid_search.best_estimator_
best_rf_classifier.fit(X, y)
feature_importances = best_rf_classifier.feature_importances_
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

