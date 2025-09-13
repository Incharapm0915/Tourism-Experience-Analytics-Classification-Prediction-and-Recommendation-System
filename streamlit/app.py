# Tourism Experience Analytics - Streamlit Dashboard
# Main Application File

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
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: #f8f9fa;
        border: 2px solid #3498db;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .recommendation-item {
        background: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .sidebar-section {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
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
            st.error("Data files not found. Please ensure Step 2 (Data Preprocessing) has been completed.")
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
        st.error(f"Error loading data: {str(e)}")
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
        st.warning("Models not found. Please ensure all modeling steps have been completed.")
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
        st.warning(f"Could not load regression model: {str(e)}")
    
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
        st.warning(f"Could not load classification model: {str(e)}")
    
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
                    st.warning(f"Could not load {model_name}: {str(model_error)}")
            
            # Load training matrix
            if os.path.exists(rec_path + 'training_matrix.csv'):
                try:
                    models['recommendation']['training_matrix'] = pd.read_csv(rec_path + 'training_matrix.csv', index_col=0)
                except Exception as matrix_error:
                    st.warning(f"Could not load training matrix: {str(matrix_error)}")
                    
    except Exception as e:
        st.warning(f"Could not load recommendation system: {str(e)}")
    
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
                return recommendations, "Using popularity-based recommendations for existing user"
            else:
                # New user - recommend most popular items overall
                item_popularity = (training_matrix > 0).sum().sort_values(ascending=False)
                recommendations = item_popularity.head(n_recommendations).index.tolist()
                return recommendations, "Using global popularity recommendations for new user"
        else:
            # No training matrix available - return sample recommendations
            sample_attractions = list(range(1, min(n_recommendations + 1, 100)))
            return sample_attractions, "Using sample recommendations (limited data available)"
            
    except Exception as e:
        # Ultimate fallback
        sample_attractions = list(range(1, n_recommendations + 1))
        return sample_attractions, f"Using fallback recommendations: {str(e)}"

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
    
    # Main header
    st.markdown('<h1 class="main-header">🏛️ Tourism Experience Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.title("Navigation")
    
    pages = {
        "🏠 Overview": "overview",
        "📊 Data Analytics": "analytics", 
        "🎯 Predictions": "predictions",
        "💡 Recommendations": "recommendations",
        "📈 Model Performance": "performance"
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
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
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(master_df):,}</h3>
            <p>Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        unique_users = master_df['UserId'].nunique() if 'UserId' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_users:,}</h3>
            <p>Unique Users</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_attractions = master_df['AttractionId'].nunique() if 'AttractionId' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{unique_attractions:,}</h3>
            <p>Unique Attractions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = master_df['Rating'].mean() if 'Rating' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_rating:.2f}</h3>
            <p>Average Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model status
    st.markdown('<h3 class="sub-header">🤖 Model Status</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'regression' in models:
            r2_score = models['regression']['metadata']['performance']['test_r2']
            st.success(f"✅ Rating Prediction Model\nR² Score: {r2_score:.3f}")
        else:
            st.error("❌ Rating Prediction Model\nNot Available")
    
    with col2:
        if 'classification' in models:
            f1_score = models['classification']['metadata']['performance']['test_f1_macro']
            st.success(f"✅ Visit Mode Classifier\nF1 Score: {f1_score:.3f}")
        else:
            st.error("❌ Visit Mode Classifier\nNot Available")
    
    with col3:
        if 'recommendation' in models:
            st.success(f"✅ Recommendation System\nAvailable")
        else:
            st.error("❌ Recommendation System\nNot Available")
    
    # Dataset information
    st.markdown('<h3 class="sub-header">📋 Dataset Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", master_df.shape)
        st.write("**Columns:**", len(master_df.columns))
        st.write("**Memory Usage:**", f"{master_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        if 'Rating' in master_df.columns:
            rating_dist = master_df['Rating'].value_counts().sort_index()
            fig = px.bar(x=rating_dist.index, y=rating_dist.values, 
                        title="Rating Distribution",
                        labels={'x': 'Rating', 'y': 'Count'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def show_analytics(master_df):
    """Data analytics and visualization page"""
    st.markdown('<h2 class="sub-header">📊 Data Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Temporal analysis
    if 'VisitYear' in master_df.columns and 'VisitMonth' in master_df.columns:
        st.markdown('<h3 class="sub-header">📅 Temporal Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Yearly trends
            yearly_data = master_df['VisitYear'].value_counts().sort_index()
            fig = px.line(x=yearly_data.index, y=yearly_data.values,
                         title="Visits by Year", markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly patterns
            monthly_data = master_df['VisitMonth'].value_counts().sort_index()
            fig = px.bar(x=monthly_data.index, y=monthly_data.values,
                        title="Visits by Month")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis
    geo_columns = ['Continent', 'Country', 'Region']
    available_geo_cols = [col for col in geo_columns if col in master_df.columns]
    
    if available_geo_cols:
        st.markdown('<h3 class="sub-header">🌍 Geographic Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Continent' in master_df.columns:
                continent_data = master_df['Continent'].value_counts()
                fig = px.pie(values=continent_data.values, names=continent_data.index,
                            title="Visits by Continent")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Country' in master_df.columns:
                country_data = master_df['Country'].value_counts().head(10)
                fig = px.bar(x=country_data.values, y=country_data.index,
                           title="Top 10 Countries", orientation='h')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Attraction analysis
    if 'AttractionType' in master_df.columns:
        st.markdown('<h3 class="sub-header">🏛️ Attraction Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            type_data = master_df['AttractionType'].value_counts().head(10)
            fig = px.bar(x=type_data.index, y=type_data.values,
                        title="Most Popular Attraction Types")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Rating' in master_df.columns:
                avg_ratings = master_df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=avg_ratings.index, y=avg_ratings.values,
                           title="Average Rating by Attraction Type")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

def show_predictions(master_df, models):
    """Prediction interface page"""
    st.markdown('<h2 class="sub-header">🎯 Attraction Rating & Visit Mode Prediction</h2>', unsafe_allow_html=True)
    
    # Input form
    st.markdown('<h3 class="sub-header">📝 Enter User Information</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Geographic Information:**")
        
        # Get unique values for dropdowns with fallbacks
        continents = master_df['Continent'].dropna().unique() if 'Continent' in master_df.columns else ['North America', 'Europe', 'Asia']
        countries = master_df['Country'].dropna().unique() if 'Country' in master_df.columns else ['USA', 'UK', 'Germany']
        
        continent = st.selectbox("Continent", continents)
        country = st.selectbox("Country", countries)
        
        st.write("**Visit Information:**")
        visit_year = st.selectbox("Visit Year", [2024, 2023, 2022, 2021, 2020])
        visit_month = st.selectbox("Visit Month", list(range(1, 13)))
    
    with col2:
        st.write("**Attraction Information:**")
        
        attraction_types = master_df['AttractionType'].dropna().unique() if 'AttractionType' in master_df.columns else ['Museum', 'Beach', 'Park']
        attraction_type = st.selectbox("Attraction Type", attraction_types)
        
        st.write("**User Behavior:**")
        user_avg_rating = st.slider("User's Average Past Rating", 1.0, 5.0, 3.5, 0.1)
        user_visit_count = st.number_input("User's Total Past Visits", min_value=1, max_value=100, value=5)
    
    # Prediction button
    if st.button("🔮 Make Predictions", type="primary"):
        # Prepare features
        user_features = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.write("**🌟 Rating Prediction:**")
            
            rating_pred, rating_info = predict_rating(models, user_features)
            if rating_pred is not None:
                st.metric("Predicted Rating", f"{rating_pred:.2f}/5.0")
                st.info(rating_info)
                
                # Rating interpretation
                if rating_pred >= 4.5:
                    st.success("Excellent experience expected!")
                elif rating_pred >= 4.0:
                    st.success("Good experience expected!")
                elif rating_pred >= 3.5:
                    st.warning("Average experience expected")
                else:
                    st.error("Below average experience expected")
            else:
                st.error(rating_info)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Visit mode prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.write("**👥 Visit Mode Prediction:**")
            
            mode_pred, mode_info, mode_probs = predict_visit_mode(models, user_features)
            if mode_pred is not None:
                st.metric("Predicted Visit Mode", mode_pred)
                st.info(mode_info)
                
                # Show probabilities if available
                if mode_probs is not None:
                    st.write("**Confidence by Mode:**")
                    class_names = ['Business', 'Couples', 'Family', 'Friends', 'Solo'][:len(mode_probs)]
                    prob_df = pd.DataFrame({
                        'Mode': class_names,
                        'Probability': mode_probs
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(prob_df, x='Mode', y='Probability',
                               title="Visit Mode Probabilities")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(mode_info)
            
            st.markdown('</div>', unsafe_allow_html=True)

def show_recommendations_page(master_df, models):
    """Recommendation interface page"""
    st.markdown('<h2 class="sub-header">💡 Personalized Attraction Recommendations</h2>', unsafe_allow_html=True)
    
    # User input
    st.markdown('<h3 class="sub-header">👤 User Selection</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User ID input
        if 'UserId' in master_df.columns:
            unique_users = sorted(master_df['UserId'].unique())
            selected_user = st.selectbox("Select User ID", unique_users[:100])  # Limit for performance
        else:
            selected_user = st.number_input("Enter User ID", min_value=1, value=1)
        
        n_recommendations = st.slider("Number of Recommendations", 5, 20, 10)
    
    with col2:
        # Show user history if available
        if 'UserId' in master_df.columns and selected_user in master_df['UserId'].values:
            user_data = master_df[master_df['UserId'] == selected_user]
            st.write(f"**User {selected_user} Statistics:**")
            st.write(f"- Total visits: {len(user_data)}")
            if 'Rating' in master_df.columns:
                st.write(f"- Average rating: {user_data['Rating'].mean():.2f}")
            if 'AttractionType' in master_df.columns:
                favorite_type = user_data['AttractionType'].mode()
                if len(favorite_type) > 0:
                    st.write(f"- Favorite attraction type: {favorite_type.iloc[0]}")
    
    # Generate recommendations
    if st.button("🎯 Get Recommendations", type="primary"):
        recommendations, rec_info = get_recommendations(models, selected_user, n_recommendations)
        
        st.markdown('<h3 class="sub-header">🎯 Recommended Attractions</h3>', unsafe_allow_html=True)
        st.info(rec_info)
        
        if recommendations:
            # Display recommendations
            for i, attraction_id in enumerate(recommendations, 1):
                st.markdown('<div class="recommendation-item">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 3, 2])
                
                with col1:
                    st.write(f"**#{i}**")
                
                with col2:
                    # Try to get attraction details
                    if 'AttractionId' in master_df.columns and attraction_id in master_df['AttractionId'].values:
                        attraction_info = master_df[master_df['AttractionId'] == attraction_id].iloc[0]
                        attraction_name = attraction_info.get('Attraction', f'Attraction {attraction_id}')
                        attraction_type = attraction_info.get('AttractionType', 'Unknown')
                        st.write(f"**{attraction_name}**")
                        st.write(f"Type: {attraction_type}")
                    else:
                        st.write(f"**Attraction {attraction_id}**")
                        st.write("Type: Unknown")
                
                with col3:
                    # Try to get rating information
                    if 'AttractionId' in master_df.columns and attraction_id in master_df['AttractionId'].values:
                        attraction_ratings = master_df[master_df['AttractionId'] == attraction_id]['Rating']
                        if len(attraction_ratings) > 0:
                            avg_rating = attraction_ratings.mean()
                            st.metric("Avg Rating", f"{avg_rating:.1f}")
                        else:
                            st.write("No ratings")
                    else:
                        st.write("Rating: N/A")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No recommendations available for this user.")

def show_performance(models):
    """Model performance page"""
    st.markdown('<h2 class="sub-header">📈 Model Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if not models:
        st.error("No models loaded. Please ensure all modeling steps have been completed.")
        return
    
    # Regression model performance
    if 'regression' in models:
        st.markdown('<h3 class="sub-header">🎯 Rating Prediction Model</h3>', unsafe_allow_html=True)
        
        reg_metadata = models['regression']['metadata']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            r2_score = reg_metadata['performance']['test_r2']
            st.metric("R² Score", f"{r2_score:.4f}")
        
        with col2:
            rmse = reg_metadata['performance']['test_rmse']
            st.metric("RMSE", f"{rmse:.4f}")
        
        with col3:
            mae = reg_metadata['performance']['test_mae']
            st.metric("MAE", f"{mae:.4f}")
        
        st.write(f"**Model:** {reg_metadata['model_name']}")
        st.write(f"**Features Used:** {reg_metadata['feature_count']}")
        st.write(f"**Training Date:** {reg_metadata['training_date']}")
    
    # Classification model performance
    if 'classification' in models:
        st.markdown('<h3 class="sub-header">👥 Visit Mode Classification Model</h3>', unsafe_allow_html=True)
        
        class_metadata = models['classification']['metadata']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = class_metadata['performance']['test_accuracy']
            st.metric("Accuracy", f"{accuracy:.4f}")
        
        with col2:
            f1_score = class_metadata['performance']['test_f1_macro']
            st.metric("F1-Score", f"{f1_score:.4f}")
        
        with col3:
            precision = class_metadata['performance']['test_precision_macro']
            st.metric("Precision", f"{precision:.4f}")
        
        with col4:
            recall = class_metadata['performance']['test_recall_macro']
            st.metric("Recall", f"{recall:.4f}")
        
        st.write(f"**Model:** {class_metadata['model_name']}")
        st.write(f"**Classes:** {class_metadata['n_classes']}")
        st.write(f"**Features Used:** {class_metadata['feature_count']}")
        st.write(f"**Training Date:** {class_metadata['training_date']}")
    
    # Recommendation system performance
    if 'recommendation' in models:
        st.markdown('<h3 class="sub-header">💡 Recommendation System Performance</h3>', unsafe_allow_html=True)
        
        rec_metadata = models['recommendation']['metadata']
        
        if 'best_method' in rec_metadata:
            best_method = rec_metadata['best_method']
            st.write(f"**Best Method:** {best_method}")
            
            if 'performance_metrics' in rec_metadata and best_method in rec_metadata['performance_metrics']:
                perf_metrics = rec_metadata['performance_metrics'][best_method]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    precision = perf_metrics.get('precision', 0)
                    st.metric("Precision@10", f"{precision:.4f}")
                
                with col2:
                    recall = perf_metrics.get('recall', 0)
                    st.metric("Recall@10", f"{recall:.4f}")
                
                with col3:
                    ndcg = perf_metrics.get('ndcg', 0)
                    st.metric("NDCG@10", f"{ndcg:.4f}")
        
        # Data statistics
        if 'data_statistics' in rec_metadata:
            data_stats = rec_metadata['data_statistics']
            st.write("**Dataset Statistics:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"- Users: {data_stats.get('n_users', 0):,}")
                st.write(f"- Attractions: {data_stats.get('n_items', 0):,}")
            
            with col2:
                st.write(f"- Interactions: {data_stats.get('n_interactions', 0):,}")
                st.write(f"- Sparsity: {data_stats.get('sparsity', 0):.1f}%")
    
    # Model comparison
    st.markdown('<h3 class="sub-header">📊 Model Comparison</h3>', unsafe_allow_html=True)
    
    comparison_data = []
    
    if 'regression' in models:
        comparison_data.append({
            'Model': 'Rating Prediction',
            'Algorithm': models['regression']['metadata']['model_name'],
            'Performance': f"R² = {models['regression']['metadata']['performance']['test_r2']:.3f}",
            'Status': 'Active'
        })
    
    if 'classification' in models:
        comparison_data.append({
            'Model': 'Visit Mode Classification',
            'Algorithm': models['classification']['metadata']['model_name'],
            'Performance': f"F1 = {models['classification']['metadata']['performance']['test_f1_macro']:.3f}",
            'Status': 'Active'
        })
    
    if 'recommendation' in models and 'metadata' in models['recommendation']:
        rec_metadata = models['recommendation']['metadata']
        if 'best_method' in rec_metadata and 'performance_metrics' in rec_metadata:
            best_method = rec_metadata['best_method']
            if best_method in rec_metadata['performance_metrics']:
                rec_perf = rec_metadata['performance_metrics'][best_method]
                comparison_data.append({
                    'Model': 'Recommendation System',
                    'Algorithm': best_method,
                    'Performance': f"NDCG = {rec_perf.get('ndcg', 0):.3f}",
                    'Status': 'Active'
                })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.info("No model performance data available for comparison.")

# Footer
def show_footer():
    """Show application footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Tourism Experience Analytics Dashboard</p>
            <p>Built with Streamlit | Machine Learning Models: Regression, Classification, Recommendation</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()
