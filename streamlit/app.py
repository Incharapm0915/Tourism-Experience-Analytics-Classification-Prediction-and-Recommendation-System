#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tourism Experience Analytics - Enhanced Streamlit Dashboard
Main Application File with Beautiful UI
"""

# Import statements - ALL AT THE TOP
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page - this uses st which is imported above
st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with more beautiful and colorful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        animation: fadeInDown 1s ease-in-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%) 1;
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        animation: slideUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .metric-card h3 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-card p {
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
        opacity: 0.95;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-card-alt {
        background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 52%, #2BFF88 90%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(43, 210, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .metric-card-alt:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(43, 210, 255, 0.6);
    }
    
    .metric-card-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4);
        transition: all 0.3s ease;
    }
    
    .metric-card-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
        transition: all 0.3s ease;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    }
    
    .recommendation-item {
        background: linear-gradient(135deg, #E3FDF5 0%, #FFE6FA 100%);
        border-left: 5px solid;
        border-image: linear-gradient(180deg, #4158D0 0%, #C850C0 50%, #FFCC70 100%) 1;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        animation: fadeInLeft 0.6s ease-out;
    }
    
    .recommendation-item:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    .sidebar-section h1, .sidebar-section h2, .sidebar-section h3 {
        color: white !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);
        }
    }
    
    .status-active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    div[data-testid="metric-container"] > div {
        color: white !important;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Loading animation */
    .loading-animation {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load processed data"""
    try:
        # Auto-detect data path
        if os.path.exists('data/processed/master_dataset.csv'):
            data_path = 'data/processed/'
        elif os.path.exists('../data/processed/master_dataset.csv'):
            data_path = '../data/processed/'
        else:
            st.error("📁 Data files not found. Please ensure Step 2 (Data Preprocessing) has been completed.")
            return None
        
        master_df = pd.read_csv(data_path + 'master_dataset.csv')
        
        # Load additional data if available
        additional_data = {}
        try:
            additional_data['user_item_matrix'] = pd.read_csv(data_path + 'user_item_matrix.csv', index_col=0)
        except:
            pass
            
        return master_df, additional_data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    
    # Auto-detect models path
    if os.path.exists('models/'):
        models_path = 'models/'
    elif os.path.exists('../models/'):
        models_path = '../models/'
    else:
        st.warning("⚠️ Models not found. Please ensure all modeling steps have been completed.")
        return {}
    
    try:
        # Load regression model
        if os.path.exists(models_path + 'regression/best_rating_predictor.pkl'):
            with open(models_path + 'regression/rating_predictor_metadata.json', 'r') as f:
                reg_metadata = json.load(f)
            models['regression'] = {
                'model': joblib.load(models_path + 'regression/best_rating_predictor.pkl'),
                'metadata': reg_metadata
            }
            # Load scaler if exists
            scaler_path = models_path + 'regression/rating_predictor_scaler.pkl'
            if os.path.exists(scaler_path):
                models['regression']['scaler'] = joblib.load(scaler_path)
    except Exception as e:
        st.warning(f"⚠️ Could not load regression model: {str(e)}")
    
    try:
        # Load classification model
        if os.path.exists(models_path + 'classification/best_visitmode_classifier.pkl'):
            with open(models_path + 'classification/visitmode_classifier_metadata.json', 'r') as f:
                class_metadata = json.load(f)
            models['classification'] = {
                'model': joblib.load(models_path + 'classification/best_visitmode_classifier.pkl'),
                'metadata': class_metadata
            }
            # Load scaler if exists
            scaler_path = models_path + 'classification/visitmode_classifier_scaler.pkl'
            if os.path.exists(scaler_path):
                models['classification']['scaler'] = joblib.load(scaler_path)
            # Load label mapping if exists
            mapping_path = models_path + 'classification/visitmode_label_mapping.pkl'
            if os.path.exists(mapping_path):
                models['classification']['label_mapping'] = joblib.load(mapping_path)
    except Exception as e:
        st.warning(f"⚠️ Could not load classification model: {str(e)}")
    
    try:
        # Load recommendation models with fallback handling
        rec_path = models_path + 'recommendation/'
        if os.path.exists(rec_path + 'recommendation_metadata.json'):
            with open(rec_path + 'recommendation_metadata.json', 'r') as f:
                rec_metadata = json.load(f)
            
            models['recommendation'] = {'metadata': rec_metadata}
            
            # Try to load individual models with error handling
            model_files = {
                'hybrid': 'hybrid_recommender.pkl',
                'user_cf': 'user_based_cf.pkl',
                'item_cf': 'item_based_cf.pkl',
                'content_cf': 'content_based_cf.pkl',
                'svd': 'svd_recommender.pkl'
            }
            
            for model_name, filename in model_files.items():
                try:
                    if os.path.exists(rec_path + filename):
                        models['recommendation'][model_name] = joblib.load(rec_path + filename)
                except Exception as model_error:
                    st.warning(f"⚠️ Could not load {model_name}: {str(model_error)}")
            
            # Load training matrix
            if os.path.exists(rec_path + 'training_matrix.csv'):
                try:
                    models['recommendation']['training_matrix'] = pd.read_csv(rec_path + 'training_matrix.csv', index_col=0)
                except Exception as matrix_error:
                    st.warning(f"⚠️ Could not load training matrix: {str(matrix_error)}")
                    
    except Exception as e:
        st.warning(f"⚠️ Could not load recommendation system: {str(e)}")
    
    return models

def predict_rating(models, user_features):
    """Predict attraction rating using regression model"""
    if 'regression' not in models:
        return None, "Regression model not available"
    
    try:
        model_info = models['regression']
        model = model_info['model']
        
        # Prepare features - use only available features
        feature_names = model_info['metadata']['features']
        
        # Create feature vector with available features, fill missing with 0
        feature_vector = {}
        for feature in feature_names:
            feature_vector[feature] = user_features.get(feature, 0)
        
        X = pd.DataFrame([feature_vector])
        
        # Apply scaling if needed
        if 'scaler' in model_info:
            X = model_info['scaler'].transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        confidence = model_info['metadata']['performance']['test_r2']
        
        return np.clip(prediction, 1, 5), f"Model confidence (R²): {confidence:.3f}"
    
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def predict_visit_mode(models, user_features):
    """Predict visit mode using classification model"""
    if 'classification' not in models:
        return None, "Classification model not available", None
    
    try:
        model_info = models['classification']
        model = model_info['model']
        
        # Prepare features - use only available features
        feature_names = model_info['metadata']['features']
        
        # Create feature vector with available features, fill missing with 0
        feature_vector = {}
        for feature in feature_names:
            feature_vector[feature] = user_features.get(feature, 0)
        
        X = pd.DataFrame([feature_vector])
        
        # Apply scaling if needed
        if 'scaler' in model_info:
            X = model_info['scaler'].transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        
        # Convert to label if mapping exists
        if 'label_mapping' in model_info and prediction in model_info['label_mapping']:
            prediction_label = model_info['label_mapping'][prediction]
        else:
            prediction_label = f"Mode {prediction}"
        
        confidence = model_info['metadata']['performance']['test_f1_macro']
        
        return prediction_label, f"Model F1-score: {confidence:.3f}", probabilities
    
    except Exception as e:
        return None, f"Prediction error: {str(e)}", None

def get_recommendations(models, user_id, n_recommendations=10):
    """Get recommendations using popularity-based approach"""
    if 'recommendation' not in models:
        return [], "Recommendation system not available"
    
    try:
        rec_models = models['recommendation']
        
        # Use popularity-based recommendations if training matrix is available
        if 'training_matrix' in rec_models:
            training_matrix = rec_models['training_matrix']
            
            # Check if user exists in training matrix
            if user_id in training_matrix.index:
                # Get items user hasn't rated
                user_ratings = training_matrix.loc[user_id]
                unrated_items = user_ratings[user_ratings == 0].index
                
                # Get popularity scores for unrated items
                item_popularity = (training_matrix > 0).sum()
                unrated_popularity = item_popularity[unrated_items].sort_values(ascending=False)
                
                recommendations = unrated_popularity.head(n_recommendations).index.tolist()
                return recommendations, "🎯 Using popularity-based recommendations for existing user"
            else:
                # New user - recommend most popular items overall
                item_popularity = (training_matrix > 0).sum().sort_values(ascending=False)
                recommendations = item_popularity.head(n_recommendations).index.tolist()
                return recommendations, "🌟 Using global popularity recommendations for new user"
        else:
            # No training matrix available - return sample recommendations
            sample_attractions = list(range(1, min(n_recommendations + 1, 100)))
            return sample_attractions, "📌 Using sample recommendations (limited data available)"
            
    except Exception as e:
        # Ultimate fallback
        sample_attractions = list(range(1, n_recommendations + 1))
        return sample_attractions, f"⚠️ Using fallback recommendations: {str(e)}"

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Load data and models
    data_result = load_data()
    if data_result is None:
        st.stop()
    
    master_df, additional_data = data_result
    models = load_models()
    
    # Main header with animation
    st.markdown('<h1 class="main-header">🏛️ Tourism Experience Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Add a beautiful separator
    st.markdown("""
        <div style="height: 2px; background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%); margin: 2rem 0; border-radius: 2px;"></div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation with enhanced styling
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<h2 style="color: white;">🎯 Navigation</h2>', unsafe_allow_html=True)
    
    pages = {
        "🏠 Overview": "overview",
        "📊 Data Analytics": "analytics", 
        "🎯 Predictions": "predictions",
        "💡 Recommendations": "recommendations",
        "📈 Model Performance": "performance"
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Add sidebar info
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<h3 style="color: white;">ℹ️ About</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
        <p style="color: white; font-size: 0.9rem;">
        This dashboard provides comprehensive tourism analytics including:
        <br>• Data visualization
        <br>• Predictive modeling
        <br>• Recommendation system
        <br>• Performance metrics
        </p>
    """, unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Page routing
    page_key = pages[selected_page]
    
    if page_key == "overview":
        show_overview(master_df, models)
    elif page_key == "analytics":
        show_analytics(master_df)
    elif page_key == "predictions":
        show_predictions(master_df, models)
    elif page_key == "recommendations":
        show_recommendations_page(master_df, models)
    elif page_key == "performance":
        show_performance(models)

def show_overview(master_df, models):
    """Overview page with key statistics"""
    st.markdown('<h2 class="sub-header">📊 Project Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics with enhanced cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(master_df):,}</h3>
            <p>📋 Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_users = master_df['UserId'].nunique() if 'UserId' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card-alt">
            <h3>{unique_users:,}</h3>
            <p>👥 Unique Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_attractions = master_df['AttractionId'].nunique() if 'AttractionId' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card-success">
            <h3>{unique_attractions:,}</h3>
            <p>🏛️ Unique Attractions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = master_df['Rating'].mean() if 'Rating' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card-warning">
            <h3>{avg_rating:.2f}</h3>
            <p>⭐ Average Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model status with enhanced styling
    st.markdown('<h3 class="sub-header">🤖 Model Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'regression' in models:
            r2_score = models['regression']['metadata']['performance']['test_r2']
            st.markdown(f"""
                <div class="info-card">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Rating Prediction Model</h4>
                    <p>R² Score: <strong>{r2_score:.3f}</strong></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card">
                    <span class="status-badge status-inactive">❌ Inactive</span>
                    <h4>Rating Prediction Model</h4>
                    <p>Not Available</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'classification' in models:
            f1_score = models['classification']['metadata']['performance']['test_f1_macro']
            st.markdown(f"""
                <div class="info-card">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Visit Mode Classifier</h4>
                    <p>F1 Score: <strong>{f1_score:.3f}</strong></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card">
                    <span class="status-badge status-inactive">❌ Inactive</span>
                    <h4>Visit Mode Classifier</h4>
                    <p>Not Available</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Fixed the error here - checking if recommendation exists without accessing 'available' key
        if 'recommendation' in models:
            st.markdown("""
                <div class="info-card">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Recommendation System</h4>
                    <p>Popularity-Based Available</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card">
                    <span class="status-badge status-inactive">❌ Inactive</span>
                    <h4>Recommendation System</h4>
                    <p>Not Available</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Dataset information with better visualization
    st.markdown('<h3 class="sub-header">📋 Dataset Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.write("**Dataset Shape:**", master_df.shape)
        st.write("**Columns:**", len(master_df.columns))
        st.write("**Memory Usage:**", f"{master_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'Rating' in master_df.columns:
            rating_dist = master_df['Rating'].value_counts().sort_index()
            
            # Create a colorful bar chart
            colors = ['#4158D0', '#C850C0', '#FFCC70', '#38ef7d', '#f5576c']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=rating_dist.index,
                    y=rating_dist.values,
                    marker=dict(
                        color=rating_dist.values,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=rating_dist.values,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Rating Distribution",
                xaxis_title="Rating",
                yaxis_title="Count",
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins"),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_analytics(master_df):
    """Data analytics and visualization page"""
    st.markdown('<h2 class="sub-header">📊 Data Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Temporal analysis
    if 'VisitYear' in master_df.columns and 'VisitMonth' in master_df.columns:
        st.markdown('<h3 class="sub-header">📅 Temporal Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Yearly trends with gradient colors
            yearly_data = master_df['VisitYear'].value_counts().sort_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_data.index,
                y=yearly_data.values,
                mode='lines+markers',
                name='Visits',
                line=dict(
                    color='#667eea',
                    width=3,
                    shape='spline'
                ),
                marker=dict(
                    size=10,
                    color=yearly_data.values,
                    colorscale='Viridis',
                    showscale=False,
                    line=dict(color='white', width=2)
                ),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig.update_layout(
                title="Visits by Year",
                xaxis_title="Year",
                yaxis_title="Number of Visits",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins"),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly patterns with colorful bars
            monthly_data = master_df['VisitMonth'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[month_names[i-1] if i <= 12 else str(i) for i in monthly_data.index],
                    y=monthly_data.values,
                    marker=dict(
                        color=monthly_data.values,
                        colorscale='Rainbow',
                        showscale=False,
                        line=dict(color='white', width=1.5)
                    ),
                    text=monthly_data.values,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Visits by Month",
                xaxis_title="Month",
                yaxis_title="Number of Visits",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis with enhanced visuals
    geo_columns = ['Continent', 'Country', 'Region']
    available_geo_cols = [col for col in geo_columns if col in master_df.columns]
    
    if available_geo_cols:
        st.markdown('<h3 class="sub-header">🌍 Geographic Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Continent' in master_df.columns:
                continent_data = master_df['Continent'].value_counts()
                
                # Create a sunburst chart for better visualization
                fig = go.Figure(data=[go.Pie(
                    labels=continent_data.index,
                    values=continent_data.values,
                    hole=.3,
                    marker=dict(
                        colors=px.colors.qualitative.Set3,
                        line=dict(color='white', width=2)
                    ),
                    textfont=dict(size=14, family="Poppins"),
                    textposition='outside',
                    textinfo='label+percent'
                )])
                
                fig.update_layout(
                    title="Visits by Continent",
                    height=400,
                    font=dict(family="Poppins"),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Country' in master_df.columns:
                country_data = master_df['Country'].value_counts().head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=country_data.values,
                        y=country_data.index,
                        orientation='h',
                        marker=dict(
                            color=country_data.values,
                            colorscale='Sunset',
                            showscale=False,
                            line=dict(color='white', width=1.5)
                        ),
                        text=country_data.values,
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Top 10 Countries",
                    xaxis_title="Number of Visits",
                    yaxis_title="Country",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins")
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Attraction analysis with improved charts
    if 'AttractionType' in master_df.columns:
        st.markdown('<h3 class="sub-header">🏛️ Attraction Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            type_data = master_df['AttractionType'].value_counts().head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=type_data.index,
                    y=type_data.values,
                    marker=dict(
                        color=type_data.values,
                        colorscale='Turbo',
                        showscale=True,
                        colorbar=dict(title="Count", thickness=15)
                    ),
                    text=type_data.values,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Most Popular Attraction Types",
                xaxis_title="Attraction Type",
                yaxis_title="Count",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins"),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Rating' in master_df.columns:
                avg_ratings = master_df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).head(10)
                
                # Create a lollipop chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=avg_ratings.values,
                    y=avg_ratings.index,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=avg_ratings.values,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Avg Rating", thickness=15),
                        line=dict(color='white', width=2)
                    )
                ))
                
                for i, (idx, val) in enumerate(avg_ratings.items()):
                    fig.add_shape(
                        type="line",
                        x0=0, x1=val,
                        y0=i, y1=i,
                        line=dict(color="rgba(102, 126, 234, 0.4)", width=2)
                    )
                
                fig.update_layout(
                    title="Average Rating by Attraction Type",
                    xaxis_title="Average Rating",
                    yaxis_title="Attraction Type",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins"),
                    xaxis=dict(range=[0, 5.5])
                )
                st.plotly_chart(fig, use_container_width=True)

def show_predictions(master_df, models):
    """Prediction interface page with enhanced UI"""
    st.markdown('<h2 class="sub-header">🎯 Attraction Rating & Visit Mode Prediction</h2>', unsafe_allow_html=True)
    
    # Add description
    st.markdown("""
        <div class="info-card">
            <p>🔮 <strong>Make predictions about tourist experiences!</strong></p>
            <p>Enter user and attraction information below to predict ratings and visit modes using our trained machine learning models.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Input form with better styling
    st.markdown('<h3 class="sub-header">📝 Enter User Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">🌍 Geographic Information</h4>', unsafe_allow_html=True)
        
        # Get unique values for dropdowns with fallbacks
        continents = master_df['Continent'].dropna().unique() if 'Continent' in master_df.columns else ['North America', 'Europe', 'Asia']
        countries = master_df['Country'].dropna().unique() if 'Country' in master_df.columns else ['USA', 'UK', 'Germany']
        
        continent = st.selectbox("Continent", continents)
        country = st.selectbox("Country", countries)
        
        st.markdown('<h4 class="gradient-text">📅 Visit Information</h4>', unsafe_allow_html=True)
        visit_year = st.selectbox("Visit Year", [2024, 2023, 2022, 2021, 2020])
        visit_month = st.selectbox("Visit Month", list(range(1, 13)))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">🏛️ Attraction Information</h4>', unsafe_allow_html=True)
        
        attraction_types = master_df['AttractionType'].dropna().unique() if 'AttractionType' in master_df.columns else ['Museum', 'Beach', 'Park']
        attraction_type = st.selectbox("Attraction Type", attraction_types)
        
        st.markdown('<h4 class="gradient-text">👤 User Behavior</h4>', unsafe_allow_html=True)
        user_avg_rating = st.slider("User's Average Past Rating", 1.0, 5.0, 3.5, 0.1)
        user_visit_count = st.number_input("User's Total Past Visits", min_value=1, max_value=100, value=5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Make Predictions", type="primary", use_container_width=True)
    
    # Prediction results
    if predict_button:
        # Prepare features
        user_features = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
        }
        
        # Show loading animation
        with st.spinner('🔄 Generating predictions...'):
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating prediction with enhanced visualization
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h4 class="gradient-text">🌟 Rating Prediction</h4>', unsafe_allow_html=True)
                
                rating_pred, rating_info = predict_rating(models, user_features)
                if rating_pred is not None:
                    # Create gauge chart for rating
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = rating_pred,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Predicted Rating"},
                        delta = {'reference': 3.0},
                        gauge = {
                            'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 2.5], 'color': '#f5576c'},
                                {'range': [2.5, 3.5], 'color': '#FFCC70'},
                                {'range': [3.5, 4.5], 'color': '#C850C0'},
                                {'range': [4.5, 5], 'color': '#38ef7d'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 4.5
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250, font={'family': "Poppins"})
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info(f"📊 {rating_info}")
                    
                    # Rating interpretation with colors
                    if rating_pred >= 4.5:
                        st.success("🎉 Excellent experience expected!")
                    elif rating_pred >= 4.0:
                        st.success("😊 Good experience expected!")
                    elif rating_pred >= 3.5:
                        st.warning("😐 Average experience expected")
                    else:
                        st.error("😕 Below average experience expected")
                else:
                    st.error(f"❌ {rating_info}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Visit mode prediction with enhanced visualization
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h4 class="gradient-text">👥 Visit Mode Prediction</h4>', unsafe_allow_html=True)
                
                mode_pred, mode_info, mode_probs = predict_visit_mode(models, user_features)
                if mode_pred is not None:
                    st.success(f"**Predicted Mode:** {mode_pred}")
                    st.info(f"📊 {mode_info}")
                    
                    # Show probabilities with enhanced chart
                    if mode_probs is not None:
                        st.markdown("**Confidence by Mode:**")
                        class_names = ['Business', 'Couples', 'Family', 'Friends', 'Solo'][:len(mode_probs)]
                        
                        # Create radial bar chart
                        fig = go.Figure()
                        
                        colors = ['#4158D0', '#C850C0', '#FFCC70', '#38ef7d', '#f5576c']
                        
                        for i, (name, prob) in enumerate(zip(class_names, mode_probs)):
                            fig.add_trace(go.Bar(
                                x=[name],
                                y=[prob],
                                name=name,
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    line=dict(color='white', width=2)
                                ),
                                text=f'{prob:.1%}',
                                textposition='outside'
                            ))
                        
                        fig.update_layout(
                            title="Visit Mode Probabilities",
                            yaxis_title="Probability",
                            height=300,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Poppins"),
                            yaxis=dict(range=[0, max(mode_probs) * 1.2])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ {mode_info}")
                
                st.markdown('</div>', unsafe_allow_html=True)
