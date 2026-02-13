# ML Assignment 2

## Problem Statement

Binary classification to predict whether a mushroom is edible or poisonous based on physical characteristics. Given the life-threatening consequences of misclassification, this problem demands high accuracy models for safe mushroom identification.

## Dataset Description

Mushroom Classification dataset from UCI Machine Learning Repository with 8,124 samples from 23 species of gilled mushrooms.

**Key characteristics:**
- **Total samples:** 8,124 mushroom instances
- **Features:** 22 categorical attributes describing physical characteristics
  - Cap attributes: cap_shape, cap_surface, cap_color
  - Physical indicators: bruises, odor
  - Gill characteristics: gill_attachment, gill_spacing, gill_size, gill_color
  - Stalk attributes: stalkshape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring
  - Veil properties: veil_type, veil_color
  - Ring features: ring_number, ring_type
  - Spore and habitat: spore_print_color, population, habitat
- **Target variable:** Mushroom_quality (binary classification)
  - 'e' = edible (4,208 samples - 51.8%)
  - 'p' = poisonous (3,916 samples - 48.2%)
- **Preprocessing:** Label encoding for all categorical features

## Models Used

Comparison table with evaluation metrics for all 6 models:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9551 | 0.9821 | 0.9598 | 0.9464 | 0.9531 | 0.9101 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9286 | 0.9506 | 0.9195 | 0.9336 | 0.9265 | 0.8572 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | 95.52% accuracy using gradient descent with sigmoid function. L2 regularization handles bias-variance tradeoff. Fast training makes it a solid baseline despite some non-linearity in features. |
| Decision Tree | Perfect 100% accuracy using information gain criterion. Hunt's algorithm with entropy-based splits achieved perfect classification, though potential overfitting warrants pruning for production use. |
| kNN | Perfect 100% accuracy with k=5 using Euclidean distance and majority voting. StandardScaler normalization essential for distance-based calculations. Lazy learning approach requires all training data at prediction time. |
| Naive Bayes | Lowest at 92.86% accuracy - violated conditional independence assumptions due to correlated mushroom features. Maximum likelihood estimation provides fast probabilistic predictions despite theoretical limitations. |
| Random Forest (Ensemble) | Perfect 100% using 100 trees with bootstrap aggregation and random feature selection. Bagging reduces variance, OOB validation confirms generalization. Parallel training makes it production-ready. |
| XGBoost (Ensemble) | Perfect 100% through sequential gradient boosting. L1/L2 regularization with learning rate 0.3 prevents overfitting. Each tree corrects residuals, reducing both bias and variance. Top choice for production. |