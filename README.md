# Feature Engineering and Selection

- Remove highly correlated features to reduce redundancy.
- Create interaction features when there are known relationships.
- Use feature importance scores to select the most relevant features.
- Apply appropriate scaling/normalization when features are on different scales.

---

# Hyperparameter Tuning

- **n_estimators**: Increase the number of trees to reduce variance (usually 100-500).
- **max_depth**: Control tree depth to prevent overfitting.
- **min_samples_split** and **min_samples_leaf**: Adjust minimum samples required to split/create leaf nodes.
- **max_features**: Modify the number of features considered at each split.
- Use cross-validation with `GridSearchCV` or `RandomizedSearchCV` for optimization.

---

# Class Imbalance Handling

- Use the `class_weight` parameter to give more importance to minority classes.
- Apply sampling techniques:
  - Oversampling minority class (e.g., SMOTE, ADASYN).
  - Undersampling majority class.
  - Combination of both (e.g., SMOTEENN, SMOTETomek).

---

# Data Quality Improvements

- Handle missing values appropriately.
- Remove or correct outliers.
- Address data leakage issues.
- Ensure proper train-test split stratification.

---

# Ensemble Methods Enhancement

- Try different bootstrapping ratios.
- Implement stacking with other models.
- Use voting classifiers to combine with other algorithms.