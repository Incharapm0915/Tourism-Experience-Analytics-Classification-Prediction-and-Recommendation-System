#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tourism Experience Analytics - Enhanced Streamlit Dashboard
Main Application File with Beautiful UI (Corrected)
"""

# Import statements - ALL AT THE TOP
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page - must be first Streamlit call
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
    * { font-family: 'Poppins', sans-serif; }
    .main-header { font-size: 3.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 2rem; padding: 1rem; animation: fadeInDown 1s ease-in-out; }
    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-30px);} to { opacity: 1; transform: translateY(0);} }
    .sub-header { font-size: 1.8rem; font-weight: 600; background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 3px solid; border-image: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%) 1; animation: fadeIn 0.8s ease-in-out; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 20px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4); transition: all 0.3s ease; animation: slideUp 0.6s ease-out; }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6); }
    .metric-card h3 { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
    .metric-card p { font-size: 1rem; font-weight: 400; margin: 0; opacity: 0.95; }
    @keyframes slideUp { from { opacity: 0; transform: translateY(20px);} to { opacity: 1; transform: translateY(0);} }
    .metric-card-alt { background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 52%, #2BFF88 90%); padding: 1.5rem; border-radius: 20px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 10px 30px rgba(43, 210, 255, 0.4); transition: all 0.3s ease; }
    .metric-card-alt:hover { transform: translateY(-5px); box-shadow: 0 15px 40px rgba(43, 210, 255, 0.6); }
    .metric-card-success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 20px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 10px 30px rgba(56, 239, 125, 0.4); transition: all 0.3s ease; }
    .metric-card-warning { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 20px; color: white; text-align: center; margin: 0.5rem 0; box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4); transition: all 0.3s ease; }
    .prediction-box { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border: none; border-radius: 20px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1); transition: all 0.3s ease; }
    .prediction-box:hover { transform: scale(1.02); box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15); }
    .recommendation-item { background: linear-gradient(135deg, #E3FDF5 0%, #FFE6FA 100%); border-left: 5px solid; border-image: linear-gradient(180deg, #4158D0 0%, #C850C0 50%, #FFCC70 100%) 1; padding: 1rem; margin: 0.8rem 0; border-radius: 15px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); transition: all 0.3s ease; animation: fadeInLeft 0.6s ease-out; }
    .recommendation-item:hover { transform: translateX(10px); box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); }
    @keyframes fadeInLeft { from { opacity: 0; transform: translateX(-20px);} to { opacity: 1; transform: translateX(0);} }
    .sidebar-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); color: white; }
    .sidebar-section h1, .sidebar-section h2, .sidebar-section h3 { color: white !important; }
    .stSelectbox > div > div { background: rgba(255, 255, 255, 0.9); border-radius: 10px; border: 2px solid rgba(102, 126, 234, 0.3); transition: all 0.3s ease; }
    .stSelectbox > div > div:hover { border-color: rgba(102, 126, 234, 0.6); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2); }
    .status-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7);} 70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0);} 100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0);} }
    .status-active { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .status-inactive { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.75rem 2rem; border-radius: 25px; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6); }
    .chart-container { background: white; border-radius: 15px; padding: 1rem; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); margin: 1rem 0; }
    div[data-testid="metric-container"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 1rem; box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3); color: white; }
    div[data-testid="metric-container"] > div { color: white !important; }
    .gradient-text { background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600; }
    .info-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08); }
    .loading-animation { display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(102, 126, 234, 0.3); border-radius: 50%; border-top-color: #667eea; animation: spin 1s ease-in-out infinite; }
    @keyframes spin { to { transform: rotate(360deg);} }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load processed data; return (master_df, additional_data) or None."""
    try:
        # Auto-detect data path
        if os.path.exists('data/processed/master_dataset.csv'):
            data_path = 'data/processed/'
        elif os.path.exists('../data/processed/master_dataset.csv'):
            data_path = '../data/processed/'
        else:
            return None
        master_df = pd.read_csv(os.path.join(data_path, 'master_dataset.csv'))
        additional_data = {}
        try:
            uim_path = os.path.join(data_path, 'user_item_matrix.csv')
            if os.path.exists(uim_path):
                additional_data['user_item_matrix'] = pd.read_csv(uim_path, index_col=0)
        except Exception:
            pass
        return master_df, additional_data
    except Exception:
        return None

@st.cache_resource
def load_models():
    """Load trained models; return dict keyed by task names."""
    models = {}
    # Auto-detect models path
    if os.path.exists('models/'):
        models_path = 'models/'
    elif os.path.exists('../models/'):
        models_path = '../models/'
    else:
        return {}

    # Regression
    try:
        reg_model_path = os.path.join(models_path, 'regression', 'best_rating_predictor.pkl')
        reg_meta_path = os.path.join(models_path, 'regression', 'rating_predictor_metadata.json')
        scaler_path = os.path.join(models_path, 'regression', 'rating_predictor_scaler.pkl')
        if os.path.exists(reg_model_path) and os.path.exists(reg_meta_path):
            with open(reg_meta_path, 'r') as f:
                reg_metadata = json.load(f)
            models['regression'] = {
                'model': joblib.load(reg_model_path),
                'metadata': reg_metadata
            }
            if os.path.exists(scaler_path):
                models['regression']['scaler'] = joblib.load(scaler_path)
    except Exception:
        pass

    # Classification
    try:
        cls_model_path = os.path.join(models_path, 'classification', 'best_visitmode_classifier.pkl')
        cls_meta_path = os.path.join(models_path, 'classification', 'visitmode_classifier_metadata.json')
        cls_scaler_path = os.path.join(models_path, 'classification', 'visitmode_classifier_scaler.pkl')
        label_map_path = os.path.join(models_path, 'classification', 'visitmode_label_mapping.pkl')
        if os.path.exists(cls_model_path) and os.path.exists(cls_meta_path):
            with open(cls_meta_path, 'r') as f:
                class_metadata = json.load(f)
            models['classification'] = {
                'model': joblib.load(cls_model_path),
                'metadata': class_metadata
            }
            if os.path.exists(cls_scaler_path):
                models['classification']['scaler'] = joblib.load(cls_scaler_path)
            if os.path.exists(label_map_path):
                models['classification']['label_mapping'] = joblib.load(label_map_path)
    except Exception:
        pass

    # Recommendation
    try:
        rec_path = os.path.join(models_path, 'recommendation')
        rec_meta_path = os.path.join(rec_path, 'recommendation_metadata.json')
        if os.path.exists(rec_meta_path):
            with open(rec_meta_path, 'r') as f:
                rec_metadata = json.load(f)
            models['recommendation'] = {'metadata': rec_metadata}
            model_files = {
                'hybrid': 'hybrid_recommender.pkl',
                'user_cf': 'user_based_cf.pkl',
                'item_cf': 'item_based_cf.pkl',
                'content_cf': 'content_based_cf.pkl',
                'svd': 'svd_recommender.pkl'
            }
            for model_name, filename in model_files.items():
                file_path = os.path.join(rec_path, filename)
                try:
                    if os.path.exists(file_path):
                        models['recommendation'][model_name] = joblib.load(file_path)
                except Exception:
                    pass
            tm_path = os.path.join(rec_path, 'training_matrix.csv')
            if os.path.exists(tm_path):
                try:
                    models['recommendation']['training_matrix'] = pd.read_csv(tm_path, index_col=0)
                except Exception:
                    pass
    except Exception:
        pass

    return models

def predict_rating(models, user_features):
    """Predict attraction rating using regression model."""
    if 'regression' not in models:
        return None, "Regression model not available"
    try:
        model_info = models['regression']
        model = model_info['model']
        feature_names = model_info.get('metadata', {}).get('features', [])
        feature_vector = {f: user_features.get(f, 0) for f in feature_names}
        X = pd.DataFrame([feature_vector]) if feature_names else pd.DataFrame([user_features])
        if 'scaler' in model_info:
            X = model_info['scaler'].transform(X)
        prediction = float(model.predict(X))
        confidence = model_info.get('metadata', {}).get('performance', {}).get('test_r2', np.nan)
        return float(np.clip(prediction, 1, 5)), f"Model confidence (R²): {confidence:.3f}" if pd.notna(confidence) else "Model confidence: N/A"
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def predict_visit_mode(models, user_features):
    """Predict visit mode using classification model."""
    if 'classification' not in models:
        return None, "Classification model not available", None
    try:
        model_info = models['classification']
        model = model_info['model']
        feature_names = model_info.get('metadata', {}).get('features', [])
        feature_vector = {f: user_features.get(f, 0) for f in feature_names}
        X = pd.DataFrame([feature_vector]) if feature_names else pd.DataFrame([user_features])
        if 'scaler' in model_info:
            X = model_info['scaler'].transform(X)
        pred_raw = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        label_map = model_info.get('label_mapping', {})
        # label_map may map int->str or str->str; handle both
        prediction_label = label_map.get(pred_raw, f"Mode {pred_raw}")
        confidence = model_info.get('metadata', {}).get('performance', {}).get('test_f1_macro', np.nan)
        return prediction_label, f"Model F1-score: {confidence:.3f}" if pd.notna(confidence) else "Model F1-score: N/A", probabilities
    except Exception as e:
        return None, f"Prediction error: {str(e)}", None

def get_recommendations(models, user_id, n_recommendations=10):
    """Get popularity-based recommendations with safe fallbacks."""
    if 'recommendation' not in models:
        return [], "Recommendation system not available"
    try:
        rec_models = models['recommendation']
        if 'training_matrix' in rec_models:
            training_matrix = rec_models['training_matrix']
            # ensure index is comparable type
            if user_id in training_matrix.index:
                user_ratings = training_matrix.loc[user_id]
                unrated_items = user_ratings[user_ratings == 0].index
                item_popularity = (training_matrix > 0).sum()
                unrated_popularity = item_popularity.loc[unrated_items].sort_values(ascending=False)
                recommendations = unrated_popularity.head(n_recommendations).index.tolist()
                return recommendations, "Using popularity-based recommendations for existing user"
            else:
                item_popularity = (training_matrix > 0).sum().sort_values(ascending=False)
                recommendations = item_popularity.head(n_recommendations).index.tolist()
                return recommendations, "Using global popularity recommendations for new user"
        else:
            sample_attractions = list(range(1, min(n_recommendations + 1, 100)))
            return sample_attractions, "Using sample recommendations (limited data)"
    except Exception as e:
        sample_attractions = list(range(1, n_recommendations + 1))
        return sample_attractions, f"Fallback recommendations: {str(e)}"

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def show_overview(master_df, models):
    st.markdown('<h2 class="sub-header">📊 Project Overview</h2>', unsafe_allow_html=True)

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
        avg_rating = float(master_df['Rating'].mean()) if 'Rating' in master_df.columns else 0.0
        st.markdown(f"""
        <div class="metric-card-warning">
            <h3>{avg_rating:.2f}</h3>
            <p>⭐ Average Rating</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-header">🤖 Model Status</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'regression' in models:
            r2_score = models['regression'].get('metadata', {}).get('performance', {}).get('test_r2', np.nan)
            r2_text = f"{r2_score:.3f}" if pd.notna(r2_score) else "N/A"
            st.markdown(f"""
                <div class="info-card">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Rating Prediction Model</h4>
                    <p>R² Score: <strong>{r2_text}</strong></p>
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
            f1_score = models['classification'].get('metadata', {}).get('performance', {}).get('test_f1_macro', np.nan)
            f1_text = f"{f1_score:.3f}" if pd.notna(f1_score) else "N/A"
            st.markdown(f"""
                <div class="info-card">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Visit Mode Classifier</h4>
                    <p>F1 Score: <strong>{f1_text}</strong></p>
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

    st.markdown('<h3 class="sub-header">📋 Dataset Information</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.write("**Dataset Shape:**", master_df.shape)
        st.write("**Columns:**", len(master_df.columns))
        try:
            mem_mb = master_df.memory_usage(deep=True).sum() / 1024**2
            st.write("**Memory Usage:**", f"{mem_mb:.2f} MB")
        except Exception:
            st.write("**Memory Usage:** N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if 'Rating' in master_df.columns:
            rating_dist = master_df['Rating'].value_counts().sort_index()
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
    st.markdown('<h2 class="sub-header">📊 Data Analytics Dashboard</h2>', unsafe_allow_html=True)

    # Temporal analysis
    if 'VisitYear' in master_df.columns and 'VisitMonth' in master_df.columns:
        st.markdown('<h3 class="sub-header">📅 Temporal Analysis</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            yearly_data = master_df['VisitYear'].value_counts().sort_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_data.index,
                y=yearly_data.values,
                mode='lines+markers',
                name='Visits',
                line=dict(color='#667eea', width=3, shape='spline'),
                marker=dict(size=10, color=yearly_data.values, colorscale='Viridis', showscale=False, line=dict(color='white', width=2)),
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
            monthly_data = master_df['VisitMonth'].value_counts().sort_index()
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            fig = go.Figure(data=[
                go.Bar(
                    x=[month_names[i-1] if 1 <= i <= 12 else str(i) for i in monthly_data.index],
                    y=monthly_data.values,
                    marker=dict(color=monthly_data.values, colorscale='Rainbow', showscale=False, line=dict(color='white', width=1.5)),
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

    # Geographic analysis
    geo_columns = ['Continent', 'Country', 'Region']
    available_geo_cols = [c for c in geo_columns if c in master_df.columns]
    if available_geo_cols:
        st.markdown('<h3 class="sub-header">🌍 Geographic Analysis</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if 'Continent' in master_df.columns:
                continent_data = master_df['Continent'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=continent_data.index,
                    values=continent_data.values,
                    hole=.3,
                    marker=dict(colors=px.colors.qualitative.Set3, line=dict(color='white', width=2)),
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
                        marker=dict(color=country_data.values, colorscale='Sunset', showscale=False, line=dict(color='white', width=1.5)),
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

    # Attraction analysis
    if 'AttractionType' in master_df.columns:
        st.markdown('<h3 class="sub-header">🏛️ Attraction Analysis</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            type_data = master_df['AttractionType'].value_counts().head(10)
            fig = go.Figure(data=[
                go.Bar(
                    x=type_data.index,
                    y=type_data.values,
                    marker=dict(color=type_data.values, colorscale='Turbo', showscale=True, colorbar=dict(title="Count", thickness=15)),
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
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=avg_ratings.values,
                    y=avg_ratings.index,
                    mode='markers',
                    marker=dict(size=15, color=avg_ratings.values, colorscale='RdYlGn', showscale=True, colorbar=dict(title="Avg Rating", thickness=15), line=dict(color='white', width=2))
                ))
                for i, (idx, val) in enumerate(avg_ratings.items()):
                    fig.add_shape(type="line", x0=0, x1=val, y0=i, y1=i, line=dict(color="rgba(102, 126, 234, 0.4)", width=2))
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
    st.markdown('<h2 class="sub-header">🎯 Attraction Rating & Visit Mode Prediction</h2>', unsafe_allow_html=True)
    st.markdown("""
        <div class="info-card">
            <p>🔮 <strong>Make predictions about tourist experiences!</strong></p>
            <p>Enter user and attraction information below to predict ratings and visit modes using trained machine learning models.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-header">📝 Enter User Information</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">🌍 Geographic Information</h4>', unsafe_allow_html=True)
        continents = master_df['Continent'].dropna().unique().tolist() if 'Continent' in master_df.columns else ['North America', 'Europe', 'Asia']
        countries = master_df['Country'].dropna().unique().tolist() if 'Country' in master_df.columns else ['USA', 'UK', 'Germany']
        continent = st.selectbox("Continent", continents, key="continent_select")
        country = st.selectbox("Country", countries, key="country_select")
        st.markdown('<h4 class="gradient-text">📅 Visit Information</h4>', unsafe_allow_html=True)
        visit_year = st.selectbox("Visit Year", [2025, 2024, 2023, 2022, 2021, 2020], index=1)
        visit_month = st.selectbox("Visit Month", list(range(1, 13)))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">🏛️ Attraction Information</h4>', unsafe_allow_html=True)
        attraction_types = master_df['AttractionType'].dropna().unique().tolist() if 'AttractionType' in master_df.columns else ['Museum', 'Beach', 'Park']
        attraction_type = st.selectbox("Attraction Type", attraction_types)
        st.markdown('<h4 class="gradient-text">👤 User Behavior</h4>', unsafe_allow_html=True)
        user_avg_rating = st.slider("User's Average Past Rating", 1.0, 5.0, 3.5, 0.1)
        user_visit_count = st.number_input("User's Total Past Visits", min_value=0, max_value=1000, value=5, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    colc1, colc2, colc3 = st.columns([1, 2, 1])
    with colc2:
        predict_button = st.button("🔮 Make Predictions", type="primary", use_container_width=True)

    if predict_button:
        user_features = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
            # If models expect encoded features, add proper encoding in preprocessing pipeline.
        }
        with st.spinner('🔄 Generating predictions...'):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h4 class="gradient-text">🌟 Rating Prediction</h4>', unsafe_allow_html=True)
                rating_pred, rating_info = predict_rating(models, user_features)
                if rating_pred is not None:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=rating_pred,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted Rating"},
                        delta={'reference': 3.0},
                        gauge={
                            'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 2.5], 'color': '#f5576c'},
                                {'range': [2.5, 4.0], 'color': '#FFCC70'},
                                {'range': [4.0, 5.0], 'color': '#38ef7d'}
                            ],
                            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': rating_pred}
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.write(rating_info)
                else:
                    st.warning(rating_info)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h4 class="gradient-text">🚶 Visit Mode Prediction</h4>', unsafe_allow_html=True)
                mode_pred, mode_info, probs = predict_visit_mode(models, user_features)
                if mode_pred is not None:
                    st.metric("Predicted Visit Mode", mode_pred)
                    st.write(mode_info)
                    if probs is not None:
                        try:
                            # Simple probability bar if class names are available
                            model_info = models.get('classification', {})
                            clf = model_info.get('model', None)
                            classes_ = getattr(clf, 'classes_', None)
                            if classes_ is not None:
                                prob_df = pd.DataFrame({'Class': classes_, 'Probability': probs})
                                fig = go.Figure(go.Bar(x=prob_df['Class'].astype(str), y=prob_df['Probability']))
                                fig.update_layout(title="Class Probabilities", yaxis=dict(range=[0,1]))
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                else:
                    st.warning(mode_info)
                st.markdown('</div>', unsafe_allow_html=True)

def show_recommendations_page(master_df, models):
    st.markdown('<h2 class="sub-header">💡 Personalized Recommendations</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><p>Get popularity-based recommendations using interaction data.</p></div>', unsafe_allow_html=True)
    # User ID input; allow str to match DataFrame index types
    user_id_input = st.text_input("Enter User ID (exact as in training matrix index)", value="1")
    n_rec = st.slider("Number of recommendations", 1, 20, 10)
    if st.button("Get Recommendations"):
        try:
            # Try int conversion but fall back to string
            try:
                candidate_id = int(user_id_input)
            except Exception:
                candidate_id = user_id_input
            items, info = get_recommendations(models, candidate_id, n_recommendations=n_rec)
            st.markdown(f'<div class="info-card"><p>{info}</p></div>', unsafe_allow_html=True)
            if items:
                for it in items:
                    st.markdown(f'<div class="recommendation-item">✅ Recommended Attraction ID: <strong>{it}</strong></div>', unsafe_allow_html=True)
            else:
                st.warning("No recommendations available.")
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")

def show_performance(models):
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"><p>Key performance metrics from held-out test sets.</p></div>', unsafe_allow_html=True)
    if 'regression' in models:
        perf = models['regression'].get('metadata', {}).get('performance', {})
        st.subheader("Rating Predictor")
        st.write({k: v for k, v in perf.items()})
    else:
        st.info("Regression model not available.")
    if 'classification' in models:
        perf = models['classification'].get('metadata', {}).get('performance', {})
        st.subheader("Visit Mode Classifier")
        st.write({k: v for k, v in perf.items()})
    else:
        st.info("Classification model not available.")
    if 'recommendation' in models:
        st.subheader("Recommendation System")
        st.write(models['recommendation'].get('metadata', {}))
    else:
        st.info("Recommendation system not available.")

def main():
    # Load data and models
    data_result = load_data()
    if data_result is None:
        st.error("📁 Data files not found. Please ensure the processed dataset exists at data/processed/master_dataset.csv or ../data/processed/master_dataset.csv.")
        st.stop()
    # Safe unpacking
    master_df, additional_data = data_result
    models = load_models()

    # Main header with animation
    st.markdown('<h1 class="main-header">🏛️ Tourism Experience Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #4158D0 0%, #C850C0 46%, #FFCC70 100%); margin: 2rem 0; border-radius: 2px;"></div>', unsafe_allow_html=True)

    # Sidebar navigation
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

if __name__ == "__main__":
    main()
