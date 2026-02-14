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
    """Inject custom CSS for unique UI appearance"""
    st.markdown("""
    <style>
        /* Custom gradient header */
        .stApp header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Custom metric cards with shadow */
        div[data-testid="stMetricValue"] {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-weight: 700;
        }
        
        /* Sidebar customization */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
            color: white;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }
        
        /* Success/error messages with custom icons */
        .element-container .stAlert {
            border-radius: 12px;
            border-left: 5px solid;
        }
        
        /* Custom button styling */
        .stDownloadButton button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            border: none;
            padding: 12px 24px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        .stDownloadButton button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        
        /* Custom table styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Animated title */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        h1 {
            animation: fadeInDown 0.8s ease-out;
        }
    </style>
    """, unsafe_allow_html=True)

def create_metric_gauge(value, title, max_val=1.0):
    """Create custom gauge chart for metrics visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#2c3e50'}},
        delta={'reference': 0.9, 'increasing': {'color': "#27ae60"}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#e74c3c'},
                {'range': [0.5, 0.75], 'color': '#f39c12'},
                {'range': [0.75, 0.9], 'color': '#3498db'},
                {'range': [0.9, 1.0], 'color': '#27ae60'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.95
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig

def create_performance_radar(metrics_dict):
    """Generate radar chart for comprehensive metric visualization"""
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='rgb(102, 126, 234)', width=3),
        marker=dict(size=8, color='rgb(118, 75, 162)'),
        name='Model Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                tickfont=dict(size=10),
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=False,
        title={
            'text': "360° Performance View",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_prediction_distribution(y_true, y_pred):
    """Visualize prediction distribution with custom styling"""
    comparison_df = pd.DataFrame({
        'Category': ['Actual Edible', 'Actual Poisonous', 'Predicted Edible', 'Predicted Poisonous'],
        'Count': [
            sum(y_true == 0),
            sum(y_true == 1),
            sum(y_pred == 0),
            sum(y_pred == 1)
        ],
        'Type': ['Actual', 'Actual', 'Predicted', 'Predicted']
    })
    
    fig = px.bar(
        comparison_df,
        x='Category',
        y='Count',
        color='Type',
        barmode='group',
        color_discrete_map={'Actual': '#3498db', 'Predicted': '#e74c3c'},
        title='Actual vs Predicted Distribution Analysis'
    )
    
    fig.update_layout(
        xaxis_title="Classification Category",
        yaxis_title="Sample Count",
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=16,
        title_font_color='#2c3e50',
        height=400,
        plot_bgcolor='rgba(245,247,250,0.5)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def main():
    """Main Streamlit application"""
    

    # Page configuration with custom theme
    st.set_page_config(
        page_title="Mushroom ML Analytics Platform",
        page_icon=" ",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS styling
    apply_custom_styling()
    
    # Custom header with emoji and gradient
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50; padding: 20px;'>
            Mushroom Classification Analytics Platform
        </h1>
        <p style='text-align: center; color: #7f8c8d; font-size: 18px; margin-bottom: 30px;'>
            <strong>Advanced Binary Classification:</strong> Distinguishing Edible from Poisonous Species
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced sidebar configuration
    with st.sidebar:
        st.markdown("<h2 style='color: white; text-align: center;'>Control Panel</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid white;'>", unsafe_allow_html=True)
        
        # Model selection with custom styling
        st.markdown("<h3 style='color: #ecf0f1;'>Step 1: Model Selection</h3>", unsafe_allow_html=True)
        selected_model_name = st.selectbox(
            "Pick your trained classifier:",
            list(MODEL_REGISTRY.keys()),
            index=0,
            help="Each model uses different algorithms and assumptions"
        )
        
        # Display model info badge
        st.markdown(f"<div style='background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; margin: 10px 0;'>" 
                   f"<p style='color: white; text-align: center; font-size: 18px; margin: 0;'>" 
                   f"<strong>{selected_model_name}</strong></p></div>", 
                   unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # File upload with custom styling
        st.markdown("<h3 style='color: #ecf0f1;'>Step 2: Data Upload</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select your test dataset (CSV format)",
            type=['csv'],
            help="Must contain 22 mushroom features + target column"
        )
        
        st.markdown("<hr style='border: 1px solid white; margin: 30px 0;'>", unsafe_allow_html=True)
        
        # Info box with custom styling
        st.markdown("""
            <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 10px; border-left: 4px solid #f39c12;'>
                <p style='color: #ecf0f1; margin: 0;'><strong>Quick Tips:</strong></p>
                <ul style='color: #bdc3c7; font-size: 14px;'>
                    <li>Ensure CSV has 22 features</li>
                    <li>Include Mushroom_quality target</li>
                    <li>Categorical values required</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<p style='color: #95a5a6; text-align: center; font-size: 12px;'>© 2026 ML Assignment 2</p>", unsafe_allow_html=True)
    

    # Main content area - Enhanced visualization
    if uploaded_file is not None:
        try:
            # Load uploaded data with progress indicator
            with st.spinner("Processing your dataset..."):
                test_data = pd.read_csv(uploaded_file)
            
            # Custom success message
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); 
                            padding: 15px; border-radius: 10px; margin: 20px 0;'>
                    <p style='color: white; font-size: 16px; margin: 0; text-align: center;'>
                        <strong>Dataset Loaded Successfully</strong> 
                        • {test_data.shape[0]} samples • {test_data.shape[1]} features
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive data preview with custom styling
            with st.expander("Explore Your Data (Interactive Preview)", expanded=False):
                col_preview1, col_preview2 = st.columns([2, 1])
                
                with col_preview1:
                    st.dataframe(
                        test_data.head(10).style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'color': '#2c3e50',
                            'border-color': '#dee2e6'
                        }),
                        use_container_width=True,
                        height=300
                    )
                
                with col_preview2:
                    st.markdown("**Dataset Statistics**")
                    st.markdown(f"- **Total Rows:** {test_data.shape[0]:,}")
                    st.markdown(f"- **Total Columns:** {test_data.shape[1]}")
                    st.markdown(f"- **Memory Usage:** {test_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
                    
                    if 'Mushroom_quality' in test_data.columns:
                        edible_count = (test_data['Mushroom_quality'] == 'e').sum()
                        poisonous_count = (test_data['Mushroom_quality'] == 'p').sum()
                        st.markdown(f"- **Edible (e):** {edible_count} ({edible_count/len(test_data)*100:.1f}%)")
                        st.markdown(f"- **Poisonous (p):** {poisonous_count} ({poisonous_count/len(test_data)*100:.1f}%)")
            
            # Verify target column exists
            if 'Mushroom_quality' not in test_data.columns:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                                padding: 20px; border-radius: 10px; margin: 20px 0;'>
                        <p style='color: white; font-size: 16px; margin: 0;'>
                            <strong>Missing Target Column</strong><br>
                            Your CSV must include 'Mushroom_quality' column for model evaluation.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                return
            
            # Preprocessing phase
            with st.spinner("Encoding categorical features..."):
                encoded_data, encoders = preprocess_uploaded_data(test_data)
                
                # Separate features and target
                feature_columns = [col for col in encoded_data.columns if col != 'Mushroom_quality']
                X_test = encoded_data[feature_columns]
                y_test = encoded_data['Mushroom_quality']
            
            # Load selected model with verification
            model_path = MODEL_REGISTRY[selected_model_name]
            
            if not os.path.exists(model_path):
                st.error(f"Model file not found at: {model_path}")
                return
            
            with st.spinner(f"Loading {selected_model_name} classifier..."):
                classifier = load_trained_classifier(model_path)
                st.success(f"{selected_model_name} loaded and ready")
            
            # Prediction phase with progress
            with st.spinner("Generating predictions..."):
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
            
            # Custom metrics dashboard with gauges
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <h2 style='text-align: center; color: #2c3e50; padding: 15px; 
                           background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           border-radius: 10px; color: white;'>
                    Performance Metrics Dashboard
                </h2>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display metrics in custom gauge format
            gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
            
            with gauge_col1:
                st.plotly_chart(
                    create_metric_gauge(metrics['Accuracy'], "Accuracy Score"),
                    use_container_width=True,
                    key="gauge_accuracy"
                )
                st.plotly_chart(
                    create_metric_gauge(metrics['Precision'], "Precision"),
                    use_container_width=True,
                    key="gauge_precision"
                )
            
            with gauge_col2:
                st.plotly_chart(
                    create_metric_gauge(metrics['AUC-ROC'], "AUC-ROC"),
                    use_container_width=True,
                    key="gauge_auc"
                )
                st.plotly_chart(
                    create_metric_gauge(metrics['Recall'], "Recall"),
                    use_container_width=True,
                    key="gauge_recall"
                )
            
            with gauge_col3:
                st.plotly_chart(
                    create_metric_gauge(metrics['F1 Score'], "F1 Score"),
                    use_container_width=True,
                    key="gauge_f1"
                )
                st.plotly_chart(
                    create_metric_gauge(metrics['Matthews Correlation'], "MCC"),
                    use_container_width=True,
                    key="gauge_mcc"
                )
            
            st.markdown("---")
            
            # Radar chart for 360° view
            st.markdown("<h3 style='text-align: center; color: #2c3e50;'>Comprehensive Performance Analysis</h3>", unsafe_allow_html=True)
            
            radar_col1, radar_col2 = st.columns([1, 1])
            
            with radar_col1:
                radar_fig = create_performance_radar(metrics)
                st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")
            
            with radar_col2:
                dist_fig = create_prediction_distribution(y_test, predictions)
                st.plotly_chart(dist_fig, use_container_width=True, key="distribution_chart")
            
            st.markdown("---")
            

            # Enhanced visualization section with tabs
            st.markdown("""
                <h2 style='text-align: center; color: #2c3e50; padding: 15px; 
                           background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           border-radius: 10px; color: white;'>
                    Detailed Classification Analysis
                </h2>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Prediction Insights"])
            
            with tab1:
                cm_col1, cm_col2 = st.columns([3, 2])
                
                with cm_col1:
                    st.markdown("### Interactive Confusion Matrix Heatmap")
                    cm_fig = render_confusion_matrix(y_test, predictions)
                    st.plotly_chart(cm_fig, use_container_width=True, key="confusion_matrix")
                
                with cm_col2:
                    st.markdown("### Matrix Interpretation")
                    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
                    
                    # Custom styled metrics boxes
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                    padding: 15px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: white; margin: 0;'>True Negatives</h4>
                            <p style='color: white; font-size: 24px; margin: 5px 0;'><strong>{tn}</strong></p>
                            <p style='color: #f8f9fa; font-size: 12px; margin: 0;'>Edible correctly identified</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 15px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: white; margin: 0;'>False Positives</h4>
                            <p style='color: white; font-size: 24px; margin: 5px 0;'><strong>{fp}</strong></p>
                            <p style='color: #f8f9fa; font-size: 12px; margin: 0;'>Edible misclassified as Poisonous</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                                    padding: 15px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: white; margin: 0;'>False Negatives</h4>
                            <p style='color: white; font-size: 24px; margin: 5px 0;'><strong>{fn}</strong></p>
                            <p style='color: #f8f9fa; font-size: 12px; margin: 0;'>Poisonous misclassified as Edible (CRITICAL!)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                    padding: 15px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: white; margin: 0;'>True Positives</h4>
                            <p style='color: white; font-size: 24px; margin: 5px 0;'><strong>{tp}</strong></p>
                            <p style='color: #f8f9fa; font-size: 12px; margin: 0;'>Poisonous correctly identified</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Safety score calculation
                    safety_score = (tn + tp) / (tn + tp + fn + fp) * 100
                    threat_level = "LOW" if fn == 0 else ("MEDIUM" if fn < 5 else "HIGH")
                    threat_color = "#27ae60" if threat_level == "LOW" else ("#f39c12" if threat_level == "MEDIUM" else "#e74c3c")
                    
                    st.markdown(f"""
                        <div style='background: {threat_color}; padding: 12px; border-radius: 8px; margin: 15px 0;'>
                            <p style='color: white; text-align: center; margin: 0;'>
                                <strong>Safety Score: {safety_score:.2f}%</strong><br>
                                <span style='font-size: 12px;'>Threat Level: {threat_level}</span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                st.markdown("### Per-Class Performance Breakdown")
                report_df = display_classification_summary(y_test, predictions)
                
                # Custom styled dataframe
                st.dataframe(
                    report_df.style.background_gradient(cmap='RdYlGn', subset=['f1-score'])
                                   .format("{:.4f}")
                                   .set_properties(**{'text-align': 'center'}),
                    use_container_width=True,
                    height=250
                )
                
                # Add interpretation guide
                st.markdown("""
                    <div style='background: #ecf0f1; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; margin-top: 20px;'>
                        <h4 style='color: #2c3e50; margin-top: 0;'>Metrics Guide</h4>
                        <ul style='color: #34495e; font-size: 14px;'>
                            <li><strong>Precision:</strong> Of all predicted as class X, how many were correct?</li>
                            <li><strong>Recall:</strong> Of all actual class X samples, how many did we find?</li>
                            <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
                            <li><strong>Support:</strong> Number of samples in each class</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### Prediction Analysis & Insights")
                
                # Calculate prediction statistics
                correct_predictions = sum(y_test == predictions)
                incorrect_predictions = sum(y_test != predictions)
                accuracy_pct = (correct_predictions / len(y_test)) * 100
                
                insight_col1, insight_col2, insight_col3 = st.columns(3)
                
                with insight_col1:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 20px; border-radius: 10px; text-align: center;'>
                            <h3 style='color: white; margin: 0;'>{correct_predictions}</h3>
                            <p style='color: #f8f9fa; margin: 5px 0;'>Correct Predictions</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with insight_col2:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 20px; border-radius: 10px; text-align: center;'>
                            <h3 style='color: white; margin: 0;'>{incorrect_predictions}</h3>
                            <p style='color: #f8f9fa; margin: 5px 0;'>Incorrect Predictions</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with insight_col3:
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                    padding: 20px; border-radius: 10px; text-align: center;'>
                            <h3 style='color: white; margin: 0;'>{accuracy_pct:.2f}%</h3>
                            <p style='color: #f8f9fa; margin: 5px 0;'>Accuracy Rate</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Model-specific insights
                if selected_model_name == "Random Forest" and hasattr(classifier, 'oob_score_'):
                    st.info(f"**Random Forest OOB Score:** {classifier.oob_score_:.4f}")
                
                if selected_model_name == "k-Nearest Neighbors (kNN)":
                    st.info(f"**kNN Configuration:** k={classifier.n_neighbors}, Distance Metric: {classifier.metric}")
                
                # Export predictions
                st.markdown("---")
                st.markdown("### Export & Download")
                
                result_df = pd.DataFrame({
                    'Actual_Class': ['Edible' if y == 0 else 'Poisonous' for y in y_test.values],
                    'Predicted_Class': ['Edible' if p == 0 else 'Poisonous' for p in predictions],
                    'Correct_Prediction': y_test.values == predictions,
                    'Actual_Encoded': y_test.values,
                    'Predicted_Encoded': predictions
                })
                
                export_col1, export_col2 = st.columns([2, 1])
                
                with export_col1:
                    st.dataframe(result_df.head(20), use_container_width=True, height=300)
                
                with export_col2:
                    csv_export = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Full Results CSV",
                        data=csv_export,
                        file_name=f"{selected_model_name.lower().replace(' ', '_')}_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style='background: #d5f4e6; padding: 15px; border-radius: 8px; border: 2px solid #27ae60;'>
                            <p style='color: #27ae60; margin: 0; text-align: center;'>
                                Ready to export {len(result_df)} predictions
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                            padding: 20px; border-radius: 10px; margin: 20px 0;'>
                    <h3 style='color: white; margin: 0;'>Processing Error</h3>
                    <p style='color: white; margin: 10px 0;'><strong>Error Details:</strong> {str(e)}</p>
                    <p style='color: #f8f9fa; font-size: 14px; margin: 0;'>
                        Please verify your CSV format matches the training data structure (22 features + target column).
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    else:

        # Enhanced landing page with custom design
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center;'>
                <h2 style='color: white; margin: 0;'>Welcome to Mushroom ML Analytics</h2>
                <p style='color: #f8f9fa; font-size: 16px; margin-top: 10px;'>
                    Upload your test dataset to begin comprehensive model evaluation
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Features showcase
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 25px; border-radius: 12px; text-align: center; min-height: 200px;'>
                    <h3 style='color: white; margin: 10px 0;'>6 ML Models</h3>
                    <p style='color: #f8f9fa; font-size: 14px;'>
                        Multiple algorithms from simple to ensemble methods
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with feature_col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 25px; border-radius: 12px; text-align: center; min-height: 200px;'>
                    <h3 style='color: white; margin: 10px 0;'>6 Metrics</h3>
                    <p style='color: #f8f9fa; font-size: 14px;'>
                        Comprehensive evaluation with accuracy, AUC, F1, and more
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with feature_col3:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                            padding: 25px; border-radius: 12px; text-align: center; min-height: 200px;'>
                    <h3 style='color: white; margin: 10px 0;'>Interactive Viz</h3>
                    <p style='color: #f8f9fa; font-size: 14px;'>
                        Gauges, radar charts, heatmaps, and distribution plots
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Available models showcase
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        padding: 25px; border-radius: 12px; margin: 20px 0;'>
                <h3 style='color: #2c3e50; text-align: center; margin-bottom: 20px;'>
                    Available Classification Models
                </h3>
        """, unsafe_allow_html=True)
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.markdown("""
                <div style='padding: 15px;'>
                    <p style='color: #2c3e50; margin: 8px 0;'>
                        <strong style='color: #667eea;'>Logistic Regression</strong><br>
                        <span style='font-size: 13px; color: #7f8c8d;'>Linear classifier with L2 regularization</span>
                    </p>
                    <p style='color: #2c3e50; margin: 8px 0;'>
                        <strong style='color: #667eea;'>Decision Tree</strong><br>
                        <span style='font-size: 13px; color: #7f8c8d;'>Hunt's algorithm with entropy criterion</span>
                    </p>
                    <p style='color: #2c3e50; margin: 8px 0;'>
                        <strong style='color: #667eea;'>k-Nearest Neighbors</strong><br>
                        <span style='font-size: 13px; color: #7f8c8d;'>Distance-based lazy learning (k=5)</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with model_col2:
            st.markdown("""
                <div style='padding: 15px;'>
                    <p style='color: #2c3e50; margin: 8px 0;'>
                        <strong style='color: #764ba2;'>Naive Bayes</strong><br>
                        <span style='font-size: 13px; color: #7f8c8d;'>Probabilistic with maximum likelihood</span>
                    </p>
                    <p style='color: #2c3e50; margin: 8px 0;'>
                        <strong style='color: #764ba2;'>Random Forest</strong><br>
                        <span style='font-size: 13px; color: #7f8c8d;'>Ensemble bagging with 100 trees</span>
                    </p>
                    <p style='color: #2c3e50; margin: 8px 0;'>
                        <strong style='color: #764ba2;'>XGBoost</strong><br>
                        <span style='font-size: 13px; color: #7f8c8d;'>Gradient boosting with L1/L2 regularization</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Requirements section
        st.markdown("""
            <div style='background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;'>
                <h4 style='color: #856404; margin-top: 0;'>Dataset Requirements</h4>
                <ul style='color: #856404; font-size: 14px;'>
                    <li><strong>Format:</strong> CSV file with comma-separated values</li>
                    <li><strong>Features:</strong> All 22 categorical mushroom attributes (cap, gill, stalk, etc.)</li>
                    <li><strong>Target:</strong> Column named 'Mushroom_quality' with values 'e' (edible) or 'p' (poisonous)</li>
                    <li><strong>Encoding:</strong> Categorical values will be automatically encoded</li>
                    <li><strong>Size:</strong> Any number of samples (tested on 8,124 training samples)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick start guide
        st.markdown("""
            <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        padding: 20px; border-radius: 10px;'>
                <h4 style='color: white; margin-top: 0;'>Quick Start Guide</h4>
                <ol style='color: white; font-size: 14px;'>
                    <li>Select your preferred ML model from the sidebar</li>
                    <li>Upload your test dataset CSV file</li>
                    <li>View comprehensive performance metrics with interactive visualizations</li>
                    <li>Analyze confusion matrix and per-class performance</li>
                    <li>Download prediction results for further analysis</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()