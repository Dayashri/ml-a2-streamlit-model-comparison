import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
import pickle
import os

def load_mushroom_data(filepath):
    """Load the mushroom dataset from CSV"""
    data = pd.read_csv(filepath)
    return data

def encode_categorical_features(dataframe):
    """Convert all categorical features to numerical using Label Encoding"""
    encoded_df = dataframe.copy()
    label_encoders = {}
   
    for column in encoded_df.columns:
        le = LabelEncoder()
        encoded_df[column] = le.fit_transform(encoded_df[column].astype(str))
        label_encoders[column] = le
   
    return encoded_df, label_encoders

def prepare_train_test_data(encoded_data, target_col='Mushroom_quality', test_ratio=0.20, random_seed=42):
    """Split data into training and testing sets"""
    feature_cols = [col for col in encoded_data.columns if col != target_col]
   
    X = encoded_data[feature_cols]
    y = encoded_data[target_col]
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )
   
    return X_train, X_test, y_train, y_test

def train_knn_model(X_train, y_train, n_neighbors=5):
    """Train K-Nearest Neighbors classifier"""
    knn_classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='uniform',
        algorithm='auto'
    )
    knn_classifier.fit(X_train, y_train)
    return knn_classifier

def calculate_performance_metrics(y_actual, y_predicted, y_probabilities):
    """Calculate all required evaluation metrics"""
    metrics_dict = {
        'Accuracy': accuracy_score(y_actual, y_predicted),
        'AUC': roc_auc_score(y_actual, y_probabilities),
        'Precision': precision_score(y_actual, y_predicted),
        'Recall': recall_score(y_actual, y_predicted),
        'F1': f1_score(y_actual, y_predicted),
        'MCC': matthews_corrcoef(y_actual, y_predicted)
    }
    return metrics_dict

def save_trained_model(model, filename='knn_model.pkl'):
    """Save the trained model to disk"""
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, filename)
   
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
   
    print(f"Model saved to: {model_path}")

def main():
    """Main execution function"""
    print("=" * 60)
    print("K-NEAREST NEIGHBORS (kNN) MODEL TRAINING")
    print("=" * 60)
   
    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'mushroom.csv')
    print(f"\nLoading data from: {data_path}")
    mushroom_data = load_mushroom_data(data_path)
    print(f"Dataset shape: {mushroom_data.shape}")
   
    # Encode categorical features
    print("\nEncoding categorical features...")
    encoded_data, encoders = encode_categorical_features(mushroom_data)
   
    # Prepare train/test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(encoded_data)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
   
    # Apply normalization (essential for distance-based methods)
    print("\nApplying feature normalization...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling completed!")
   
    # Train model
    print("\nTraining K-Nearest Neighbors model...")
    trained_model = train_knn_model(X_train_scaled, y_train)
    print("Training completed!")
   
    # Make predictions
    print("\nGenerating predictions...")
    y_pred = trained_model.predict(X_test_scaled)
    y_prob = trained_model.predict_proba(X_test_scaled)[:, 1]
   
    # Calculate metrics
    print("\nCalculating performance metrics...")
    performance_metrics = calculate_performance_metrics(y_test, y_pred, y_prob)
   
    # Display results
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    for metric_name, metric_value in performance_metrics.items():
        print(f"{metric_name:<15}: {metric_value:.4f}")
    print("=" * 60)
   
    # Save model
    print("\nSaving trained model...")
    save_trained_model(trained_model)
   
    print("\n✓ K-Nearest Neighbors model training complete!\n")
   
    return performance_metrics

if __name__ == "__main__":
    metrics = main()