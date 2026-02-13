# ML Assignment 2

## Problem Statement

This project addresses a critical binary classification challenge in mycology and food safety. The objective is to predict whether a mushroom is edible or poisonous based on its physical characteristics. Given the life-threatening consequences of misclassification, this problem demands high accuracy and reliability. By analyzing features like cap shape, color, odor, and habitat, we build predictive models that can assist in mushroom identification and contribute to foraging safety.

## Dataset Description

I'm working with the Mushroom Classification dataset from the UCI Machine Learning Repository. This dataset contains descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms from the Agaricus and Lepiota families.

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
- **Data quality:** All features are categorical, requiring label encoding for model training
- **Class distribution:** Well-balanced dataset with approximately equal representation

## Models Used

Comparison table with evaluation metrics for all 6 models:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.9552 | 0.9825 | 0.9568 | 0.9499 | 0.9534 | 0.9103 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kNN | - | - | - | - | - | - |
| Naive Bayes | - | - | - | - | - | - |
| Random Forest (Ensemble) | - | - | - | - | - | - |
| XGBoost (Ensemble) | - | - | - | - | - | - |

## Model Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | [Analysis to be added] |
| Decision Tree | [Analysis to be added] |
| kNN | [Analysis to be added] |
| Naive Bayes | [Analysis to be added] |
| Random Forest (Ensemble) | [Analysis to be added] |
| XGBoost (Ensemble) | [Analysis to be added] |