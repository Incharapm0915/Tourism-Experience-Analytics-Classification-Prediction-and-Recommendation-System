#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tourism Experience Analytics - Enhanced Streamlit Dashboard
Fixed version for GitHub deployment with proper data loading
"""

# Import statements
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'colorful'

# Enhanced theme definitions
THEMES = {
    'dark': {
        'name': '🌙 Dark Mode',
        'bg': 'linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0e4b99 100%)',
        'sidebar_bg': 'rgba(26, 26, 46, 0.95)',
        'card_bg': 'rgba(30, 33, 57, 0.95)',
        'text': '#ffffff',
        'text_secondary': '#b0b7c3',
        'primary': '#bb86fc',
        'secondary': '#03dac6',
        'accent': '#cf6679',
        'gradient1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
        'gradient3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'gradient4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
        'shadow': '0 20px 60px rgba(0, 0, 0, 0.6)',
        'hover_shadow': '0 30px 80px rgba(187, 134, 252, 0.5)',
        'glow': '0 0 30px rgba(187, 134, 252, 0.3)',
        'border': 'rgba(255, 255, 255, 0.1)'
    },
    'light': {
        'name': '☀️ Light Mode',
        'bg': 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 25%, #a8edea 50%, #fed6e3 75%, #d299c2 100%)',
        'sidebar_bg': 'rgba(248, 249, 250, 0.95)',
        'card_bg': 'rgba(255, 255, 255, 0.95)',
        'text': '#2d3436',
        'text_secondary': '#636e72',
        'primary': '#4285f4',
        'secondary': '#34a853',
        'accent': '#ea4335',
        'gradient1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient2': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        'gradient3': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
        'gradient4': 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
        'shadow': '0 20px 60px rgba(0, 0, 0, 0.15)',
        'hover_shadow': '0 30px 80px rgba(66, 133, 244, 0.4)',
        'glow': '0 0 30px rgba(66, 133, 244, 0.2)',
        'border': 'rgba(0, 0, 0, 0.1)'
    },
    'colorful': {
        'name': '🎨 Colorful',
        'bg': 'linear-gradient(135deg, #667eea 0%, #764ba2 20%, #f093fb 40%, #4facfe 60%, #00f2fe 80%, #43e97b 100%)',
        'sidebar_bg': 'rgba(255, 255, 255, 0.95)',
        'card_bg': 'rgba(255, 255, 255, 0.95)',
        'text': '#2d3436',
        'text_secondary': '#636e72',
        'primary': '#6c5ce7',
        'secondary': '#a29bfe',
        'accent': '#fd79a8',
        'gradient1': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
        'gradient2': 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
        'gradient3': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
        'gradient4': 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
        'shadow': '0 20px 60px rgba(108, 92, 231, 0.4)',
        'hover_shadow': '0 30px 80px rgba(108, 92, 231, 0.6)',
        'glow': '0 0 30px rgba(108, 92, 231, 0.4)',
        'border': 'rgba(108, 92, 231, 0.2)'
    },
    'midnight': {
        'name': '🌌 Midnight',
        'bg': 'linear-gradient(135deg, #0f0c29 0%, #302b63 25%, #24243e 50%, #1a1a2e 75%, #16213e 100%)',
        'sidebar_bg': 'rgba(36, 36, 62, 0.95)',
        'card_bg': 'rgba(48, 43, 99, 0.8)',
        'text': '#e0e0e0',
        'text_secondary': '#b0b0b0',
        'primary': '#7f5af0',
        'secondary': '#2cb67d',
        'accent': '#ff6b6b',
        'gradient1': 'linear-gradient(135deg, #7f5af0 0%, #2cb67d 100%)',
        'gradient2': 'linear-gradient(135deg, #ff6b6b 0%, #feca57 100%)',
        'gradient3': 'linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%)',
        'gradient4': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
        'shadow': '0 20px 60px rgba(127, 90, 240, 0.4)',
        'hover_shadow': '0 30px 80px rgba(127, 90, 240, 0.6)',
        'glow': '0 0 30px rgba(127, 90, 240, 0.4)',
        'border': 'rgba(127, 90, 240, 0.2)'
    },
    'ocean': {
        'name': '🌊 Ocean',
        'bg': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 25%, #1e90ff 50%, #87ceeb 75%, #b0e0e6 100%)',
        'sidebar_bg': 'rgba(42, 82, 152, 0.95)',
        'card_bg': 'rgba(255, 255, 255, 0.15)',
        'text': '#ffffff',
        'text_secondary': '#e0e0e0',
        'primary': '#00d2ff',
        'secondary': '#3a7bd5',
        'accent': '#0575e6',
        'gradient1': 'linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%)',
        'gradient2': 'linear-gradient(135deg, #0575e6 0%, #021b79 100%)',
        'gradient3': 'linear-gradient(135deg, #00f260 0%, #0575e6 100%)',
        'gradient4': 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
        'shadow': '0 20px 60px rgba(0, 210, 255, 0.4)',
        'hover_shadow': '0 30px 80px rgba(0, 210, 255, 0.6)',
        'glow': '0 0 30px rgba(0, 210, 255, 0.4)',
        'border': 'rgba(0, 210, 255, 0.2)'
    }
}

def get_theme_css():
    """Generate enhanced CSS based on selected theme"""
    theme = THEMES[st.session_state.theme]
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');
        
        * {{
            font-family: 'Inter', 'Poppins', sans-serif;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .stApp {{
            background: {theme['bg']};
            background-attachment: fixed;
            color: {theme['text']};
            background-size: 400% 400%;
            animation: gradientShift 20s ease infinite;
        }}
        
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            25% {{ background-position: 100% 50%; }}
            50% {{ background-position: 100% 100%; }}
            75% {{ background-position: 0% 100%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .main-header {{
            font-family: 'Playfair Display', serif;
            font-size: 4.5rem;
            font-weight: 700;
            background: {theme['gradient1']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            padding: 3rem 2rem;
            animation: fadeInDown 1.2s ease-in-out, float 8s ease-in-out infinite;
            text-shadow: 0 5px 15px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }}
        
        .main-header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            animation: shimmer 3s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ left: -100%; }}
            100% {{ left: 100%; }}
        }}
        
        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-50px) scale(0.9); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-15px); }}
        }}
        
        .sub-header {{
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            font-weight: 600;
            background: {theme['gradient2']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 3rem 0 2rem 0;
            padding-bottom: 1rem;
            border-bottom: 3px solid;
            border-image: {theme['gradient2']} 1;
            animation: slideInLeft 1s ease-out;
            position: relative;
        }}
        
        @keyframes slideInLeft {{
            from {{ opacity: 0; transform: translateX(-100px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        .metric-card {{
            background: {theme['gradient1']};
            padding: 2.5rem;
            border-radius: 30px;
            color: white;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: {theme['shadow']};
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.8s ease-out;
            position: relative;
            overflow: hidden;
            border: 1px solid {theme['border']};
            backdrop-filter: blur(20px);
        }}
        
        .metric-card:hover {{
            transform: translateY(-15px) scale(1.05);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .metric-card h3 {{
            font-size: 3.5rem;
            font-weight: 800;
            margin: 1rem 0;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
            font-family: 'Poppins', sans-serif;
        }}
        
        .metric-card p {{
            font-size: 1.2rem;
            font-weight: 500;
            margin: 0;
            opacity: 0.95;
            position: relative;
            z-index: 1;
            letter-spacing: 1px;
        }}
        
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(50px) scale(0.9); }}
            to {{ opacity: 1; transform: translateY(0) scale(1); }}
        }}
        
        .metric-card-alt {{
            background: {theme['gradient2']};
            padding: 2.5rem;
            border-radius: 30px;
            color: white;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: {theme['shadow']};
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            border: 1px solid {theme['border']};
            backdrop-filter: blur(20px);
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }}
        
        .metric-card-alt:hover {{
            transform: translateY(-15px) scale(1.05);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .metric-card-success {{
            background: {theme['gradient3']};
            padding: 2.5rem;
            border-radius: 30px;
            color: white;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: {theme['shadow']};
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            border: 1px solid {theme['border']};
            backdrop-filter: blur(20px);
            animation: fadeInUp 0.8s ease-out 0.4s both;
        }}
        
        .metric-card-success:hover {{
            transform: translateY(-15px) scale(1.05);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .metric-card-warning {{
            background: {theme['gradient4']};
            padding: 2.5rem;
            border-radius: 30px;
            color: white;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: {theme['shadow']};
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            border: 1px solid {theme['border']};
            backdrop-filter: blur(20px);
            animation: fadeInUp 0.8s ease-out 0.6s both;
        }}
        
        .metric-card-warning:hover {{
            transform: translateY(-15px) scale(1.05);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .info-card {{
            background: {theme['card_bg']};
            border-radius: 25px;
            padding: 2.5rem;
            margin: 2rem 0;
            box-shadow: {theme['shadow']};
            backdrop-filter: blur(30px);
            border: 1px solid {theme['border']};
            transition: all 0.4s ease;
            animation: fadeIn 1s ease-out;
            position: relative;
            overflow: hidden;
        }}
        
        .info-card:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .prediction-box {{
            background: {theme['card_bg']};
            border-radius: 30px;
            padding: 3rem;
            margin: 2rem 0;
            box-shadow: {theme['shadow']};
            backdrop-filter: blur(30px);
            border: 2px solid {theme['border']};
            transition: all 0.4s ease;
            animation: scaleIn 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
        }}
        
        @keyframes scaleIn {{
            from {{ opacity: 0; transform: scale(0.8); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        
        .prediction-box:hover {{
            transform: scale(1.03);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .recommendation-item {{
            background: {theme['card_bg']};
            border-left: 6px solid;
            border-image: {theme['gradient3']} 1;
            padding: 2rem;
            margin: 1.5rem 0;
            border-radius: 20px;
            box-shadow: {theme['shadow']};
            transition: all 0.4s ease;
            animation: slideInRight 0.8s ease-out;
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }}
        
        @keyframes slideInRight {{
            from {{ opacity: 0; transform: translateX(100px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        .recommendation-item:hover {{
            transform: translateX(15px) scale(1.02);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
            border-left-width: 10px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 0.8rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1rem;
            animation: pulse 3s infinite;
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
        }}
        
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7); }}
            70% {{ box-shadow: 0 0 0 20px rgba(255, 255, 255, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }}
        }}
        
        .status-active {{
            background: {theme['gradient3']};
            color: white;
        }}
        
        .status-inactive {{
            background: {theme['gradient2']};
            color: white;
        }}
        
        .stButton > button {{
            background: {theme['gradient1']};
            color: white !important;
            border: none;
            padding: 1rem 3rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.2rem;
            letter-spacing: 1px;
            transition: all 0.4s ease;
            box-shadow: {theme['shadow']};
            text-transform: uppercase;
            position: relative;
            overflow: hidden;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-5px) scale(1.05);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .gradient-text {{
            background: {theme['gradient1']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
        }}
        
        .css-1d391kg {{
            background: {theme['sidebar_bg']};
            backdrop-filter: blur(30px);
            border-right: 1px solid {theme['border']};
        }}
        
        .chart-container {{
            background: {theme['card_bg']};
            border-radius: 25px;
            padding: 2rem;
            box-shadow: {theme['shadow']};
            margin: 2rem 0;
            backdrop-filter: blur(20px);
            border: 1px solid {theme['border']};
            transition: all 0.3s ease;
        }}
        
        .chart-container:hover {{
            transform: translateY(-5px);
            box-shadow: {theme['hover_shadow']};
        }}
        
        ::-webkit-scrollbar {{
            width: 12px;
            height: 12px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {theme['gradient1']};
            border-radius: 10px;
            border: 2px solid transparent;
            background-clip: content-box;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {theme['gradient2']};
            background-clip: content-box;
        }}
        
        @media (max-width: 768px) {{
            .main-header {{
                font-size: 3rem;
                padding: 2rem 1rem;
            }}
            
            .sub-header {{
                font-size: 2rem;
            }}
            
            .metric-card h3 {{
                font-size: 2.5rem;
            }}
            
            .metric-card, .metric-card-alt, .metric-card-success, .metric-card-warning {{
                padding: 2rem;
            }}
            
            .info-card, .prediction-box {{
                padding: 2rem;
            }}
        }}
    </style>
    """

# Apply theme CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

def create_theme_switcher():
    """Create enhanced theme switcher"""
    cols = st.columns(len(THEMES))
    for idx, (key, theme) in enumerate(THEMES.items()):
        with cols[idx]:
            if st.button(
                theme['name'], 
                key=f"theme_{key}", 
                use_container_width=True,
                help=f"Switch to {theme['name']} theme"
            ):
                st.session_state.theme = key
                st.rerun()

def create_sample_data():
    """Create sample tourism data for demo purposes."""
    np.random.seed(42)
    
    n_users = 1000
    n_attractions = 200
    n_records = 5000
    
    data = {
        'UserId': np.random.randint(1, n_users + 1, n_records),
        'AttractionId': np.random.randint(1, n_attractions + 1, n_records),
        'Rating': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.1, 0.25, 0.35, 0.25]),
        'VisitYear': np.random.choice([2020, 2021, 2022, 2023, 2024], n_records),
        'VisitMonth': np.random.randint(1, 13, n_records),
        'Continent': np.random.choice(['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'], 
                                    n_records, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]),
        'Country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Canada', 'Italy'], 
                                  n_records),
        'AttractionType': np.random.choice(['Museum', 'Beach', 'Park', 'Monument', 'Zoo', 'Gallery', 'Castle', 'Garden'], 
                                         n_records),
        'UserAvgRating': np.random.uniform(2.0, 5.0, n_records),
        'UserVisitCount': np.random.randint(1, 50, n_records)
    }
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    """Load data with comprehensive fallback system."""
    
    # Try exact paths based on GitHub structure
    data_paths = [
        'data/processed/master_dataset.csv',
        'data/master_dataset.csv', 
        'master_dataset.csv',
        'processed/master_dataset.csv'
    ]
    
    master_df = None
    data_path = None
    
    # Try to load from known paths
    for path in data_paths:
        try:
            if os.path.exists(path):
                master_df = pd.read_csv(path)
                data_path = os.path.dirname(path) if os.path.dirname(path) else '.'
                st.success(f"✅ Data loaded from: {path}")
                break
        except Exception as e:
            st.warning(f"Failed to load from {path}: {e}")
            continue
    
    # If no exact match, search for any CSV that might be the dataset
    if master_df is None:
        search_dirs = ['.', 'data', 'data/processed', 'processed']
        
        for directory in search_dirs:
            if not os.path.exists(directory):
                continue
                
            try:
                files = os.listdir(directory)
                csv_files = [f for f in files if f.endswith('.csv')]
                
                # Look for likely dataset files
                candidates = [f for f in csv_files if any(keyword in f.lower() 
                            for keyword in ['master', 'dataset', 'tourism', 'data'])]
                
                if candidates:
                    file_path = os.path.join(directory, candidates[0])
                    master_df = pd.read_csv(file_path)
                    data_path = directory
                    st.warning(f"⚠️ Using alternative dataset: {file_path}")
                    break
            except Exception:
                continue
    
    # If still no data, show debug info and create sample data
    if master_df is None:
        st.error("❌ No data files found. Using sample data for demonstration.")
        
        with st.expander("🔍 Debug Information"):
            st.write("**Current directory:**", os.getcwd())
            st.write("**Files in current directory:**")
            try:
                for item in sorted(os.listdir('.')):
                    if os.path.isdir(item):
                        st.write(f"📁 {item}/")
                        if item == 'data' and os.path.exists('data'):
                            try:
                                data_files = os.listdir('data')
                                for f in sorted(data_files):
                                    if os.path.isdir(os.path.join('data', f)):
                                        st.write(f"  📁 data/{f}/")
                                        if f == 'processed':
                                            proc_files = os.listdir('data/processed')
                                            for pf in sorted(proc_files):
                                                st.write(f"    📄 data/processed/{pf}")
                                    else:
                                        st.write(f"  📄 data/{f}")
                            except:
                                pass
                    else:
                        st.write(f"📄 {item}")
            except Exception as e:
                st.write(f"Error: {e}")
        
        master_df = create_sample_data()
        data_path = None
    
    # Load additional data
    additional_data = {}
    if data_path and data_path != '.':
        uim_paths = [
            os.path.join(data_path, 'user_item_matrix.csv'),
            'user_item_matrix.csv'
        ]
        
        for uim_path in uim_paths:
            try:
                if os.path.exists(uim_path):
                    additional_data['user_item_matrix'] = pd.read_csv(uim_path, index_col=0)
                    st.info(f"✅ User-item matrix loaded from: {uim_path}")
                    break
            except:
                pass
    
    return master_df, additional_data

@st.cache_resource
def load_models():
    """Load models with comprehensive fallback system."""
    models = {}
    
    model_paths = ['models/', '../models/', './models/']
    models_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            models_path = path
            st.info(f"📁 Models found at: {path}")
            break
    
    if models_path is None:
        st.warning("⚠️ No models directory found. Using basic functionality.")
        return {}
    
    # Load regression model
    try:
        reg_model = os.path.join(models_path, 'regression', 'best_rating_predictor.pkl')
        reg_meta = os.path.join(models_path, 'regression', 'rating_predictor_metadata.json')
        reg_scaler = os.path.join(models_path, 'regression', 'rating_predictor_scaler.pkl')
        
        if os.path.exists(reg_model) and os.path.exists(reg_meta):
            with open(reg_meta, 'r') as f:
                reg_metadata = json.load(f)
            models['regression'] = {
                'model': joblib.load(reg_model),
                'metadata': reg_metadata
            }
            if os.path.exists(reg_scaler):
                models['regression']['scaler'] = joblib.load(reg_scaler)
            st.success("✅ Regression model loaded")
        else:
            st.warning("⚠️ Regression model files not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading regression model: {e}")
    
    # Load classification model
    try:
        cls_model = os.path.join(models_path, 'classification', 'best_visitmode_classifier.pkl')
        cls_meta = os.path.join(models_path, 'classification', 'visitmode_classifier_metadata.json')
        cls_scaler = os.path.join(models_path, 'classification', 'visitmode_classifier_scaler.pkl')
        cls_labels = os.path.join(models_path, 'classification', 'visitmode_label_mapping.pkl')
        
        if os.path.exists(cls_model) and os.path.exists(cls_meta):
            with open(cls_meta, 'r') as f:
                class_metadata = json.load(f)
            models['classification'] = {
                'model': joblib.load(cls_model),
                'metadata': class_metadata
            }
            if os.path.exists(cls_scaler):
                models['classification']['scaler'] = joblib.load(cls_scaler)
            if os.path.exists(cls_labels):
                models['classification']['label_mapping'] = joblib.load(cls_labels)
            st.success("✅ Classification model loaded")
        else:
            st.warning("⚠️ Classification model files not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading classification model: {e}")
    
    # Load recommendation models
    try:
        rec_path = os.path.join(models_path, 'recommendation')
        rec_meta = os.path.join(rec_path, 'recommendation_metadata.json')
        
        if os.path.exists(rec_meta):
            with open(rec_meta, 'r') as f:
                rec_metadata = json.load(f)
            models['recommendation'] = {'metadata': rec_metadata}
            
            model_files = {
                'hybrid': 'hybrid_recommender.pkl',
                'user_cf': 'user_based_cf.pkl', 
                'item_cf': 'item_based_cf.pkl',
                'content_cf': 'content_based_cf.pkl',
                'svd': 'svd_recommender.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                file_path = os.path.join(rec_path, filename)
                try:
                    if os.path.exists(file_path):
                        models['recommendation'][model_name] = joblib.load(file_path)
                        loaded_count += 1
                except Exception:
                    pass
            
            tm_path = os.path.join(rec_path, 'training_matrix.csv')
            if os.path.exists(tm_path):
                try:
                    models['recommendation']['training_matrix'] = pd.read_csv(tm_path, index_col=0)
                except Exception:
                    pass
            
            if loaded_count > 0:
                st.success(f"✅ Recommendation system loaded ({loaded_count} models)")
            else:
                st.warning("⚠️ No recommendation model files found")
        else:
            st.warning("⚠️ Recommendation metadata not found")
    except Exception as e:
        st.warning(f"⚠️ Error loading recommendation models: {e}")
    
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
        prediction = float(model.predict(X)[0])
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
        pred_raw = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
        label_map = model_info.get('label_mapping', {})
        prediction_label = label_map.get(str(pred_raw), label_map.get(pred_raw, f"Mode {pred_raw}"))
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

def show_overview(master_df, models):
    st.markdown('<h2 class="sub-header">📊 Project Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="animation-delay: 0.1s;">
            <h3>{len(master_df):,}</h3>
            <p>📋 Total Records</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        unique_users = master_df['UserId'].nunique() if 'UserId' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card-alt" style="animation-delay: 0.2s;">
            <h3>{unique_users:,}</h3>
            <p>👥 Unique Users</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        unique_attractions = master_df['AttractionId'].nunique() if 'AttractionId' in master_df.columns else 0
        st.markdown(f"""
        <div class="metric-card-success" style="animation-delay: 0.3s;">
            <h3>{unique_attractions:,}</h3>
            <p>🏛️ Unique Attractions</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_rating = float(master_df['Rating'].mean()) if 'Rating' in master_df.columns else 0.0
        st.markdown(f"""
        <div class="metric-card-warning" style="animation-delay: 0.4s;">
            <h3>{avg_rating:.2f}</h3>
            <p>⭐ Average Rating</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h3 class="sub-header">🤖 Model Performance Dashboard</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'regression' in models:
            r2_score = models['regression'].get('metadata', {}).get('performance', {}).get('test_r2', np.nan)
            r2_text = f"{r2_score:.3f}" if pd.notna(r2_score) else "N/A"
            r2_percent = r2_score * 100 if pd.notna(r2_score) else 0
            st.markdown(f"""
                <div class="info-card" style="animation-delay: 0.5s;">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Rating Prediction Model</h4>
                    <p style="font-size: 1.3rem; margin: 1rem 0;">R² Score: <strong>{r2_text}</strong></p>
                    <div style="margin-top: 1.5rem;">
                        <div style="background: rgba(255,255,255,0.2); border-radius: 15px; height: 15px; overflow: hidden;">
                            <div style="width: {r2_percent}%; background: {THEMES[st.session_state.theme]['gradient3']}; height: 100%; border-radius: 15px; transition: width 2s ease;"></div>
                        </div>
                        <p style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">Performance: {r2_percent:.1f}%</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card" style="animation-delay: 0.5s;">
                    <span class="status-badge status-inactive">❌ Inactive</span>
                    <h4>Rating Prediction Model</h4>
                    <p>Not Available</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'classification' in models:
            f1_score = models['classification'].get('metadata', {}).get('performance', {}).get('test_f1_macro', np.nan)
            f1_text = f"{f1_score:.3f}" if pd.notna(f1_score) else "N/A"
            f1_percent = f1_score * 100 if pd.notna(f1_score) else 0
            st.markdown(f"""
                <div class="info-card" style="animation-delay: 0.6s;">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Visit Mode Classifier</h4>
                    <p style="font-size: 1.3rem; margin: 1rem 0;">F1 Score: <strong>{f1_text}</strong></p>
                    <div style="margin-top: 1.5rem;">
                        <div style="background: rgba(255,255,255,0.2); border-radius: 15px; height: 15px; overflow: hidden;">
                            <div style="width: {f1_percent}%; background: {THEMES[st.session_state.theme]['gradient1']}; height: 100%; border-radius: 15px; transition: width 2s ease;"></div>
                        </div>
                        <p style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">Performance: {f1_percent:.1f}%</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card" style="animation-delay: 0.6s;">
                    <span class="status-badge status-inactive">❌ Inactive</span>
                    <h4>Visit Mode Classifier</h4>
                    <p>Not Available</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'recommendation' in models:
            st.markdown(f"""
                <div class="info-card" style="animation-delay: 0.7s;">
                    <span class="status-badge status-active">✅ Active</span>
                    <h4 class="gradient-text">Recommendation System</h4>
                    <p style="font-size: 1.3rem; margin: 1rem 0;">Popularity-Based Engine</p>
                    <div style="margin-top: 1.5rem;">
                        <div style="background: rgba(255,255,255,0.2); border-radius: 15px; height: 15px; overflow: hidden;">
                            <div style="width: 95%; background: {THEMES[st.session_state.theme]['gradient2']}; height: 100%; border-radius: 15px; transition: width 2s ease;"></div>
                        </div>
                        <p style="text-align: center; margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.8;">✨ Ready to suggest attractions</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-card" style="animation-delay: 0.7s;">
                    <span class="status-badge status-inactive">❌ Inactive</span>
                    <h4>Recommendation System</h4>
                    <p>Not Available</p>
                </div>
            """, unsafe_allow_html=True)

def show_analytics(master_df):
    st.markdown('<h2 class="sub-header">📊 Advanced Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    theme = THEMES[st.session_state.theme]
    
    if 'VisitYear' in master_df.columns and 'VisitMonth' in master_df.columns:
        st.markdown('<h3 class="sub-header">📅 Temporal Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            yearly_data = master_df['VisitYear'].value_counts().sort_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_data.index,
                y=yearly_data.values,
                mode='lines+markers',
                name='Visits',
                line=dict(color=theme['primary'], width=6, shape='spline'),
                marker=dict(size=15, color=yearly_data.values, colorscale='Viridis'),
                fill='tozeroy'
            ))
            
            fig.update_layout(
                title="📈 Yearly Visit Trends",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Rating' in master_df.columns:
                rating_dist = master_df['Rating'].value_counts().sort_index()
                
                fig = go.Figure(data=[go.Pie(
                    labels=[f"⭐ {rating}" for rating in rating_dist.index],
                    values=rating_dist.values,
                    hole=.6,
                    marker=dict(colors=['#ff6b6b', '#feca57', '#48dbfb', '#0abde3', '#00d2d3'])
                )])
                
                fig.update_layout(
                    title="⭐ Rating Distribution",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_predictions(master_df, models):
    st.markdown('<h2 class="sub-header">🎯 AI-Powered Predictions</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="info-card" style="background: {THEMES[st.session_state.theme]['gradient1']}; color: white; border: none;">
            <h3 style="margin-bottom: 1rem; color: white;">🔮 Prediction Engine</h3>
            <p style="font-size: 1.2rem; line-height: 1.6;">Enter user and attraction details to get AI-powered predictions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        continents = master_df['Continent'].dropna().unique().tolist() if 'Continent' in master_df.columns else ['North America', 'Europe', 'Asia']
        continent = st.selectbox("🌐 Continent", continents)
        visit_year = st.selectbox("📆 Visit Year", [2025, 2024, 2023, 2022])
        visit_month = st.selectbox("🗓️ Visit Month", list(range(1, 13)))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        attraction_types = master_df['AttractionType'].dropna().unique().tolist() if 'AttractionType' in master_df.columns else ['Museum', 'Beach', 'Park']
        attraction_type = st.selectbox("🎭 Attraction Type", attraction_types)
        user_avg_rating = st.slider("⭐ User's Average Rating", 1.0, 5.0, 3.5, 0.1)
        user_visit_count = st.number_input("📊 Total Past Visits", min_value=0, value=5)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔮 Generate Predictions", type="primary", use_container_width=True):
        user_features = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<h4>🌟 Rating Prediction</h4>')
            
            rating_pred, rating_info = predict_rating(models, user_features)
            if rating_pred is not None:
                st.metric("Predicted Rating", f"{rating_pred:.2f}/5.0")
                st.info(rating_info)
            else:
                st.warning(rating_info)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<h4>🚶 Visit Mode Prediction</h4>')
            
            mode_pred, mode_info, probs = predict_visit_mode(models, user_features)
            if mode_pred is not None:
                st.metric("Predicted Visit Mode", mode_pred)
                st.info(mode_info)
            else:
                st.warning(mode_info)
            st.markdown('</div>', unsafe_allow_html=True)

def show_recommendations_page(master_df, models):
    st.markdown('<h2 class="sub-header">💡 Smart Recommendations</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        user_id_input = st.text_input("👤 Enter User ID", value="1")
        n_rec = st.slider("📊 Number of Recommendations", 1, 20, 10)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            try:
                candidate_id = int(user_id_input) if user_id_input.isdigit() else user_id_input
                items, info = get_recommendations(models, candidate_id, n_recommendations=n_rec)
                
                st.info(info)
                
                if items:
                    st.markdown('<h3>✨ Recommended Attractions</h3>')
                    for idx, item in enumerate(items):
                        st.markdown(f"""
                            <div class="recommendation-item">
                                <h4>Attraction #{item}</h4>
                                <p>Confidence: {95 - idx * 2}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No recommendations available.")
            except Exception as e:
                st.error(f"Error: {e}")

def show_performance(models):
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Rating Predictor", "🎯 Visit Mode Classifier", "💡 Recommendation System"])
    
    with tab1:
        if 'regression' in models:
            perf = models['regression'].get('metadata', {}).get('performance', {})
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            if perf:
                for metric, value in perf.items():
                    if isinstance(value, (int, float)):
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Regression model not available.")
    
    with tab2:
        if 'classification' in models:
            perf = models['classification'].get('metadata', {}).get('performance', {})
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            if perf:
                for metric, value in perf.items():
                    if isinstance(value, (int, float)):
                        st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Classification model not available.")
    
    with tab3:
        if 'recommendation' in models:
            metadata = models['recommendation'].get('metadata', {})
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            if metadata:
                for key, value in metadata.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Recommendation system not available.")

def main():
    data_result = load_data()
    if data_result is None:
        st.error("Data files not found. Please ensure the processed dataset exists.")
        st.stop()
    
    master_df, additional_data = data_result
    models = load_models()
    
    st.markdown('<h1 class="main-header">🏛️ Tourism Experience Analytics</h1>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="text-align: center; margin-bottom: 2rem;">🎨 Choose Your Theme</h3>', unsafe_allow_html=True)
    create_theme_switcher()
    
    with st.sidebar:
        st.markdown(f"""
            <div style="background: {THEMES[st.session_state.theme]['gradient1']}; 
                        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;">
                <h2 style="color: white; margin: 0;">🎯 Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        pages = {
            "🏠 Overview": "overview",
            "📊 Analytics": "analytics", 
            "🎯 Predictions": "predictions",
            "💡 Recommendations": "recommendations",
            "📈 Performance": "performance"
        }
        
        selected_page = st.selectbox("Navigate to:", list(pages.keys()), label_visibility="collapsed")
        
        # Debug mode
        if st.checkbox("🔍 Debug Mode"):
            st.write("**Current directory:**", os.getcwd())
            st.write("**Files:**")
            for item in sorted(os.listdir('.')):
                if os.path.isdir(item):
                    st.write(f"📁 {item}/")
                else:
                    st.write(f"📄 {item}")
    
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
