import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Model configurations
MODEL_REGISTRY = {
    'Logistic Regression': 'model/logistic_regression_model.pkl',
    'Decision Tree': 'model/decision_tree_model.pkl',
    'k-Nearest Neighbors (kNN)': 'model/knn_model.pkl',
    'Naive Bayes': 'model/naive_bayes_model.pkl',
    'Random Forest': 'model/random_forest_model.pkl',
    'XGBoost': 'model/xgboost_model.pkl'
}

def load_trained_classifier(model_path):
    """Load a pickled model from disk"""
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
    return classifier

def preprocess_uploaded_data(dataframe):
    """Encode categorical features for prediction"""
    processed_df = dataframe.copy()
    label_encoders = {}
    
    for column in processed_df.columns:
        le = LabelEncoder()
        processed_df[column] = le.fit_transform(processed_df[column].astype(str))
        label_encoders[column] = le
    
    return processed_df, label_encoders

def compute_evaluation_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive performance metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'Matthews Correlation': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def render_confusion_matrix(y_true, y_pred, class_labels=['Edible', 'Poisonous']):
    """Generate interactive confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create annotated heatmap
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        width=500,
        height=500
    )
    
    return fig

def display_classification_summary(y_true, y_pred):
    """Generate detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=['Edible', 'Poisonous'], output_dict=True)
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    return report_df

def apply_custom_styling():
    """Minimal styling for clean appearance"""

    pass  # Using default Streamlit styling

# Removed fancy gauge charts - using simple st.metric() instead

def create_prediction_distribution(y_true, y_pred):
    """Visualize prediction distribution"""
    comparison_df = pd.DataFrame({
        'Mushroom Type': ['Edible', 'Edible', 'Poisonous', 'Poisonous'],
        'Count': [
            sum(y_true == 0),
            sum(y_pred == 0),
            sum(y_true == 1),
            sum(y_pred == 1)
        ],
        'Type': ['Actual', 'Predicted', 'Actual', 'Predicted']
    })
    
    fig = px.bar(
        comparison_df,
        x='Mushroom Type',
        y='Count',
        color='Type',
        barmode='group',
        title='Actual vs Predicted Distribution',
        color_discrete_map={'Actual': '#636EFA', 'Predicted': '#EF553B'}
    )
    
    fig.update_layout(
        xaxis_title="Mushroom Classification",
        yaxis_title="Sample Count",
        height=400
    )
    
    return fig

def main():
    """Main Streamlit application"""
    

    # Page configuration
    st.set_page_config(
        page_title="Mushroom Classification",
        page_icon=" ",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Simple header
    st.title("Mushroom Classification Model Evaluation")
    st.write("Binary Classification: Edible vs Poisonous Mushrooms")
    st.markdown("---")
    

    # Sidebar configuration
    with st.sidebar:
        st.header("Control Panel")
        
        # Model selection
        st.subheader(" Step 1: Model Selection")
        selected_model_name = st.selectbox(
            "Pick your trained classifier:",
            list(MODEL_REGISTRY.keys()),
            index=0
        )
        
        st.info(f"Selected: **{selected_model_name}**")
        
        # File upload
        st.subheader("Step 2: Data Upload")
        uploaded_file = st.file_uploader(
            "Select your test dataset (CSV format)",
            type=['csv']
        )
        
        st.markdown("---")
        
        # Info box
        st.info("""
            **Quick Tips:**
            - Ensure CSV has 22 features
            - Include Mushroom_quality target
            - Categorical values required
        """)
    

    # Main content area
    if uploaded_file is not None:
        try:
            # Load uploaded data
            test_data = pd.read_csv(uploaded_file)
            st.success(f"Dataset Loaded: {test_data.shape[0]} samples, {test_data.shape[1]} features")
            
            # Verify required column exists
            if 'Mushroom_quality' not in test_data.columns:
                st.error("Missing Target Column: Your CSV must include 'Mushroom_quality' column for model evaluation.")
                return
            
            # Preprocessing phase
            encoded_data, encoders = preprocess_uploaded_data(test_data)
            
            # Separate features and target
            feature_columns = [col for col in encoded_data.columns if col != 'Mushroom_quality']
            X_test = encoded_data[feature_columns]
            y_test = encoded_data['Mushroom_quality']
            
            # Load selected model
            model_path = MODEL_REGISTRY[selected_model_name]
            
            if not os.path.exists(model_path):
                st.error(f"Model file not found at: {model_path}")
                return
            
            classifier = load_trained_classifier(model_path)
            
            # Generate predictions
            # Apply StandardScaler normalization for kNN (required for distance calculations)
            if selected_model_name == 'k-Nearest Neighbors (kNN)':
                scaler = StandardScaler()
                X_test_scaled = scaler.fit_transform(X_test)
                predictions = classifier.predict(X_test_scaled)
                
                # Handle probability scores
                if hasattr(classifier, 'predict_proba'):
                    probabilities = classifier.predict_proba(X_test_scaled)[:, 1]
                else:
                    probabilities = predictions
            else:
                predictions = classifier.predict(X_test)
                
                # Handle probability scores
                if hasattr(classifier, 'predict_proba'):
                    probabilities = classifier.predict_proba(X_test)[:, 1]
                else:
                    probabilities = predictions
            
            # Calculate comprehensive metrics
            metrics = compute_evaluation_metrics(y_test, predictions, probabilities)
            

            # Performance Metrics Dashboard
            st.header("Performance Metrics")
            
            # Display metrics in simple format
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            
            with metric_col2:
                st.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            
            with metric_col3:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                st.metric("Matthews Correlation", f"{metrics['Matthews Correlation']:.4f}")
            
            st.markdown("---")
            

            # Distribution analysis
            st.subheader("Prediction Distribution Analysis")
            dist_fig = create_prediction_distribution(y_test, predictions)
            st.plotly_chart(dist_fig, use_container_width=True, key="distribution_chart")
            
            st.markdown("---")
            
            # Detailed analysis
            st.header("Detailed Classification Analysis")
            
            tab1, tab2 = st.tabs(["Confusion Matrix", "Classification Report"])
            
            with tab1:
                cm_col1, cm_col2 = st.columns([3, 2])
                
                with cm_col1:
                    st.markdown("### Interactive Confusion Matrix Heatmap")
                    cm_fig = render_confusion_matrix(y_test, predictions)
                    st.plotly_chart(cm_fig, use_container_width=True, key="confusion_matrix")
                
                with cm_col2:

                    st.markdown("### Matrix Interpretation")
                    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                    
                    # Simple metrics display
                    st.metric("True Negatives", tn, help="Edible correctly identified")
                    st.metric("False Positives", fp, help="Edible misclassified as Poisonous")
                    st.metric("False Negatives", fn, help="Poisonous misclassified as Edible (CRITICAL!)", delta=None if fn == 0 else f"-{fn}")
                    st.metric("True Positives", tp, help="Poisonous correctly identified")
                    
                    # Safety score
                    safety_score = (tn + tp) / (tn + tp + fn + fp) * 100
                    threat_level = "LOW" if fn == 0 else ("MEDIUM" if fn < 5 else "HIGH")
                    
                    if threat_level == "LOW":
                        st.success(f"Safety Score: {safety_score:.2f}% | Threat Level: {threat_level}")
                    elif threat_level == "MEDIUM":
                        st.warning(f"Safety Score: {safety_score:.2f}% | Threat Level: {threat_level}")
                    else:
                        st.error(f"Safety Score: {safety_score:.2f}% | Threat Level: {threat_level}")
            
            with tab2:

                st.markdown("### Per-Class Performance Breakdown")
                report_df = display_classification_summary(y_test, predictions)
                
                # Display dataframe
                st.dataframe(
                    report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score'])
                                   .format("{:.4f}")
                                   .set_properties(**{'text-align': 'center'}),
                    use_container_width=True,
                    height=250
                )
                
                # Interpretation guide
                st.info("""
                    **Metrics Guide**
                    - **Precision:** Of all predicted as class X, how many were correct?
                    - **Recall:** Of all actual class X samples, how many did we find?
                    - **F1-Score:** Harmonic mean of precision and recall
                    - **Support:** Number of samples in each class
                """)
                
                # CSV Export
                st.markdown("---")
                st.subheader("Export Predictions")
                
                result_df = pd.DataFrame({
                    'Actual_Class': ['Edible' if y == 0 else 'Poisonous' for y in y_test.values],
                    'Predicted_Class': ['Edible' if p == 0 else 'Poisonous' for p in predictions],
                    'Correct': y_test.values == predictions
                })
                
                st.dataframe(result_df.head(10), use_container_width=True)
                
                csv_export = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_export,
                    file_name=f"predictions_{selected_model_name.lower().replace(' ', '_')}.csv",
                    mime='text/csv'
                )
            
        except Exception as e:

            st.error(f"**Processing Error:** {str(e)}")
            st.warning("Please verify your CSV format matches the training data structure (22 features + target column).")
    
    else:
        # Landing page
        st.info("**Welcome to Mushroom ML Analytics** - Upload your test dataset to begin comprehensive model evaluation")
        
        # Features showcase
        st.subheader("Features")
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("**6 ML Models**")
            st.write("Multiple algorithms from simple to ensemble methods")
        
        with feature_col2:
            st.markdown("**6 Metrics**")
            st.write("Comprehensive evaluation with accuracy, AUC, F1, and more")
        
        with feature_col3:
            st.markdown("**Interactive Visualizations**")
            st.write("Charts, heatmaps, and distribution plots")
        
        # Available models
        st.subheader("Available Classification Models")
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.markdown("""
                - **Logistic Regression** - Linear classifier with L2 regularization
                - **Decision Tree** - Hunt's algorithm with entropy criterion
                - **k-Nearest Neighbors** - Distance-based lazy learning (k=5)
            """)
        
        with model_col2:
            st.markdown("""
                - **Naive Bayes** - Probabilistic with maximum likelihood
                - **Random Forest** - Ensemble bagging with 100 trees
                - **XGBoost** - Gradient boosting with L1/L2 regularization
            """)
        
        # Requirements
        st.warning("""
            **Dataset Requirements**
            - **Format:** CSV file with comma-separated values
            - **Features:** All 22 categorical mushroom attributes
            - **Target:** Column named 'Mushroom_quality' with values 'e' (edible) or 'p' (poisonous)
            - **Encoding:** Categorical values will be automatically encoded
        """)

if __name__ == "__main__":
    main()