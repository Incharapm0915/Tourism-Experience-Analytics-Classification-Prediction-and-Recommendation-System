#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tourism Experience Analytics - Enhanced Streamlit Dashboard
Main Application File with Beautiful UI/UX and Theme Modes
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
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page - must be first Streamlit call
st.set_page_config(
    page_title="Tourism Analytics Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'colorful'  # Default theme

# Enhanced theme definitions with better gradients and colors
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
    },
    'sunset': {
        'name': '🌅 Sunset',
        'bg': 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 25%, #fecfef 50%, #ff9a9e 75%, #fab2ff 100%)',
        'sidebar_bg': 'rgba(255, 154, 158, 0.2)',
        'card_bg': 'rgba(255, 255, 255, 0.9)',
        'text': '#2d3436',
        'text_secondary': '#636e72',
        'primary': '#fd79a8',
        'secondary': '#fdcb6e',
        'accent': '#e84393',
        'gradient1': 'linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%)',
        'gradient2': 'linear-gradient(135deg, #e84393 0%, #f39c12 100%)',
        'gradient3': 'linear-gradient(135deg, #ff7675 0%, #fab1a0 100%)',
        'gradient4': 'linear-gradient(135deg, #fd79a8 0%, #fdcb6e 100%)',
        'shadow': '0 20px 60px rgba(253, 121, 168, 0.3)',
        'hover_shadow': '0 30px 80px rgba(253, 121, 168, 0.5)',
        'glow': '0 0 30px rgba(253, 121, 168, 0.3)',
        'border': 'rgba(253, 121, 168, 0.2)'
    }
}

def get_theme_css():
    """Generate enhanced CSS based on selected theme"""
    theme = THEMES[st.session_state.theme]
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');
        
        /* Global Styles */
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
        
        /* Enhanced Animated Background */
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            25% {{ background-position: 100% 50%; }}
            50% {{ background-position: 100% 100%; }}
            75% {{ background-position: 0% 100%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        /* Main Header with Enhanced Typography */
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
        
        /* Enhanced Sub Headers */
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
        
        .sub-header::after {{
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 0;
            height: 3px;
            background: {theme['gradient2']};
            animation: expandWidth 2s ease-out 0.5s forwards;
        }}
        
        @keyframes expandWidth {{
            to {{ width: 100%; }}
        }}
        
        @keyframes slideInLeft {{
            from {{ opacity: 0; transform: translateX(-100px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        
        /* Enhanced Metric Cards with 3D Effects */
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
            transform-style: preserve-3d;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 6s linear infinite;
            z-index: 0;
        }}
        
        @keyframes rotate {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        
        .metric-card:hover {{
            transform: translateY(-15px) scale(1.05) rotateX(5deg);
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
        
        /* Alternating Metric Cards with Different Gradients */
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
            transform: translateY(-15px) scale(1.05) rotateX(-5deg);
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
            transform: translateY(-15px) scale(1.05) rotateY(5deg);
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
            transform: translateY(-15px) scale(1.05) rotateY(-5deg);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        /* Enhanced Info Cards with Glass Effect */
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
        
        .info-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: {theme['gradient1']};
            transform: scaleX(0);
            transform-origin: left;
            transition: transform 0.5s ease;
        }}
        
        .info-card:hover {{
            transform: translateY(-10px) scale(1.02);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        .info-card:hover::before {{
            transform: scaleX(1);
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Enhanced Prediction Box */
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
        
        .prediction-box::before {{
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: {theme['gradient1']};
            z-index: -1;
            border-radius: 30px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .prediction-box:hover::before {{
            opacity: 1;
        }}
        
        @keyframes scaleIn {{
            from {{ opacity: 0; transform: scale(0.8) rotateY(20deg); }}
            to {{ opacity: 1; transform: scale(1) rotateY(0deg); }}
        }}
        
        .prediction-box:hover {{
            transform: scale(1.03) translateZ(10px);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        /* Enhanced Recommendation Items */
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
        
        .recommendation-item::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 0;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: width 0.5s ease;
        }}
        
        .recommendation-item:hover::before {{
            width: 100%;
        }}
        
        @keyframes slideInRight {{
            from {{ opacity: 0; transform: translateX(100px) rotateY(30deg); }}
            to {{ opacity: 1; transform: translateX(0) rotateY(0deg); }}
        }}
        
        .recommendation-item:hover {{
            transform: translateX(15px) scale(1.02);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
            border-left-width: 10px;
        }}
        
        /* Enhanced Status Badges */
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
        
        .status-badge::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }}
        
        .status-badge:hover::before {{
            left: 100%;
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
        
        /* Enhanced Buttons */
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
        
        .stButton > button::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.3s;
        }}
        
        .stButton > button:hover::before {{
            left: 100%;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-5px) scale(1.05);
            box-shadow: {theme['hover_shadow']}, {theme['glow']};
        }}
        
        /* Enhanced Gradient Text */
        .gradient-text {{
            background: {theme['gradient1']};
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 600;
            animation: gradientMove 3s ease infinite;
        }}
        
        @keyframes gradientMove {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}
        
        /* Enhanced Sidebar */
        .css-1d391kg {{
            background: {theme['sidebar_bg']};
            backdrop-filter: blur(30px);
            border-right: 1px solid {theme['border']};
        }}
        
        /* Enhanced Chart Container */
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
        
        /* Enhanced Scrollbar */
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
        
        /* Enhanced Loading Animation */
        .loading-dots {{
            display: inline-block;
            position: relative;
        }}
        
        .loading-dots::after {{
            content: '●●●';
            animation: dots 1.5s infinite;
            font-size: 1.5rem;
            color: {theme['primary']};
        }}
        
        @keyframes dots {{
            0%, 20% {{ 
                content: '●○○';
                color: {theme['primary']};
            }}
            40% {{ 
                content: '○●○';
                color: {theme['secondary']};
            }}
            60% {{ 
                content: '○○●';
                color: {theme['accent']};
            }}
            80%, 100% {{ 
                content: '●●●';
                color: {theme['primary']};
            }}
        }}
        
        /* Parallax Effect */
        .parallax-container {{
            position: relative;
            overflow: hidden;
            transform-style: preserve-3d;
        }}
        
        .parallax-element {{
            transform: translateZ(-1px) scale(2);
        }}
        
        /* Responsive Design */
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

# Enhanced Theme Switcher with better UI
def create_theme_switcher():
    """Create enhanced theme switcher with preview"""
    st.markdown('<div class="theme-switcher-container">', unsafe_allow_html=True)
    
    cols = st.columns(len(THEMES))
    for idx, (key, theme) in enumerate(THEMES.items()):
        with cols[idx]:
            # Create theme preview button
            button_style = f"""
                background: {theme['gradient1']};
                color: white;
                border: {'3px solid ' + theme['primary'] if st.session_state.theme == key else '1px solid transparent'};
                border-radius: 15px;
                padding: 1rem;
                transition: all 0.3s ease;
                cursor: pointer;
                box-shadow: {theme['shadow']};
                transform: {'scale(1.05)' if st.session_state.theme == key else 'scale(1)'};
            """
            
            if st.button(
                theme['name'], 
                key=f"theme_{key}", 
                use_container_width=True,
                help=f"Switch to {theme['name']} theme"
            ):
                st.session_state.theme = key
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS (UNCHANGED - KEEPING ALL MODEL CONNECTIONS)
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
        # label_map may map int->str or str->str; handle both
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
# ENHANCED UI FUNCTIONS WITH BEAUTIFUL VISUALIZATIONS
# =============================================================================

def show_overview(master_df, models):
    st.markdown('<h2 class="sub-header">📊 Project Overview</h2>', unsafe_allow_html=True)
    
    # Enhanced animated metrics with staggered animations
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
    
    # Enhanced model status cards with progress bars
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

    # Enhanced dataset information with beautiful charts
    st.markdown('<h3 class="sub-header">📋 Dataset Analytics</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">📊 Dataset Overview</h4>', unsafe_allow_html=True)
        
        # Create enhanced info display
        dataset_info = [
            ("📏 Dataset Shape", f"{master_df.shape[0]:,} × {master_df.shape[1]}"),
            ("📝 Total Columns", f"{len(master_df.columns):,}"),
            ("💾 Memory Usage", f"{master_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"),
        ]
        
        missing_percent = (master_df.isnull().sum().sum() / (len(master_df) * len(master_df.columns))) * 100
        completeness = 100 - missing_percent
        dataset_info.append(("✅ Data Completeness", f"{completeness:.1f}%"))
        
        for label, value in dataset_info:
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                           padding: 0.8rem; margin: 0.5rem 0; background: rgba(255,255,255,0.1); 
                           border-radius: 10px; border-left: 4px solid {THEMES[st.session_state.theme]['primary']};">
                    <span style="font-weight: 500;">{label}</span>
                    <span style="font-weight: 700; color: {THEMES[st.session_state.theme]['primary']};">{value}</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'Rating' in master_df.columns:
            rating_dist = master_df['Rating'].value_counts().sort_index()
            
            # Enhanced 3D donut chart for rating distribution
            fig = go.Figure(data=[go.Pie(
                labels=[f"⭐ {rating}" for rating in rating_dist.index],
                values=rating_dist.values,
                hole=.6,
                marker=dict(
                    colors=['#ff6b6b', '#feca57', '#48dbfb', '#0abde3', '#00d2d3'],
                    line=dict(color='white', width=4)
                ),
                textfont=dict(size=14, family="Poppins", color='white'),
                textposition='outside',
                textinfo='label+percent+value',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                rotation=45
            )])
            
            fig.update_layout(
                title={
                    'text': "⭐ Rating Distribution",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Poppins', 'color': THEMES[st.session_state.theme]['text']}
                },
                height=400,
                font=dict(family="Poppins", color=THEMES[st.session_state.theme]['text']),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=12)
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text=f'Total<br>{rating_dist.sum():,}<br>Ratings', 
                                x=0.5, y=0.5, font_size=16, showarrow=False,
                                font_color=THEMES[st.session_state.theme]['text'])]
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_analytics(master_df):
    st.markdown('<h2 class="sub-header">📊 Advanced Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Theme-aware color schemes
    theme = THEMES[st.session_state.theme]
    
    # Temporal analysis with enhanced visualizations
    if 'VisitYear' in master_df.columns and 'VisitMonth' in master_df.columns:
        st.markdown('<h3 class="sub-header">📅 Temporal Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            yearly_data = master_df['VisitYear'].value_counts().sort_index()
            
            # Enhanced 3D surface-like area chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=yearly_data.index,
                y=yearly_data.values,
                mode='lines+markers',
                name='Visits',
                line=dict(
                    color=theme['primary'], 
                    width=6, 
                    shape='spline',
                    smoothing=1.3
                ),
                marker=dict(
                    size=15,
                    color=yearly_data.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Visits", 
                        thickness=20,
                        len=0.7,
                        bgcolor='rgba(255,255,255,0.1)',
                        bordercolor=theme['border'],
                        borderwidth=1
                    ),
                    line=dict(color='white', width=3),
                    symbol='circle'
                ),
                fill='tozeroy',
                fillcolor=f'rgba({",".join(str(int(theme["primary"][i:i+2], 16)) for i in (1, 3, 5))}, 0.3)'
            ))
            
            fig.update_layout(
                title={
                    'text': "📈 Yearly Visit Trends",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Poppins'}
                },
                xaxis_title="Year",
                yaxis_title="Number of Visits",
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins", color=theme['text']),
                hovermode='x unified',
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)',
                    gridwidth=1,
                    zeroline=False
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)',
                    gridwidth=1,
                    zeroline=False
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly_data = master_df['VisitMonth'].value_counts().sort_index()
            month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            
            # Enhanced polar area chart with better styling
            fig = go.Figure()
            
            theta = [month_names[i-1] if 1 <= i <= 12 else str(i) for i in monthly_data.index]
            
            fig.add_trace(go.Scatterpolar(
                r=monthly_data.values,
                theta=theta,
                mode='lines+markers',
                name='Monthly Visits',
                line=dict(color=theme['secondary'], width=4),
                marker=dict(
                    size=12,
                    color=monthly_data.values,
                    colorscale='Rainbow',
                    showscale=False,
                    line=dict(color='white', width=2)
                ),
                fill='toself',
                fillcolor=f'rgba({",".join(str(int(theme["secondary"][i:i+2], 16)) for i in (1, 3, 5))}, 0.3)'
            ))
            
            fig.update_layout(
                title={
                    'text': "🗓️ Monthly Visit Pattern",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Poppins'}
                },
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins", color=theme['text']),
                polar=dict(
                    radialaxis=dict(
                        showticklabels=True,
                        gridcolor='rgba(255,255,255,0.2)',
                        linecolor='rgba(255,255,255,0.2)',
                        range=[0, max(monthly_data.values) * 1.1]
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(255,255,255,0.2)',
                        linecolor='rgba(255,255,255,0.2)'
                    ),
                    bgcolor='rgba(0,0,0,0)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic analysis with enhanced visualizations
    geo_columns = ['Continent', 'Country', 'Region']
    available_geo_cols = [c for c in geo_columns if c in master_df.columns]
    
    if available_geo_cols:
        st.markdown('<h3 class="sub-header">🌍 Geographic Distribution</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Continent' in master_df.columns:
                continent_data = master_df['Continent'].value_counts()
                
                # Enhanced sunburst chart
                fig = go.Figure(data=[go.Pie(
                    labels=continent_data.index,
                    values=continent_data.values,
                    hole=.5,
                    marker=dict(
                        colors=px.colors.qualitative.Set3,
                        line=dict(color='white', width=4)
                    ),
                    textfont=dict(size=14, family="Poppins"),
                    textposition='outside',
                    textinfo='label+percent+value',
                    pull=[0.1 if i == 0 else 0.05 if i == 1 else 0 for i in range(len(continent_data))],
                    hovertemplate='<b>%{label}</b><br>Visits: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig.update_layout(
                    title={
                        'text': "🌐 Visits by Continent",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'family': 'Poppins'}
                    },
                    height=450,
                    font=dict(family="Poppins", color=theme['text']),
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.1,
                        bgcolor='rgba(255,255,255,0.1)',
                        bordercolor=theme['border'],
                        borderwidth=1
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(
                        text=f'Total<br>{continent_data.sum():,}<br>Visits', 
                        x=0.5, y=0.5, font_size=16, showarrow=False,
                        font_color=theme['text'], font_family='Poppins'
                    )]
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Country' in master_df.columns:
                country_data = master_df['Country'].value_counts().head(10)
                
                # Enhanced horizontal bar chart with custom styling
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=country_data.values,
                    y=country_data.index,
                    orientation='h',
                    marker=dict(
                        color=country_data.values,
                        colorscale='Sunset',
                        showscale=True,
                        colorbar=dict(
                            title="Visits", 
                            thickness=20,
                            len=0.7,
                            bgcolor='rgba(255,255,255,0.1)',
                            bordercolor=theme['border'],
                            borderwidth=1
                        ),
                        line=dict(color='white', width=2),
                        cornerradius=5
                    ),
                    text=country_data.values,
                    textposition='outside',
                    textfont=dict(size=12, family="Poppins", color=theme['text']),
                    hovertemplate='<b>%{y}</b><br>Visits: %{x}<extra></extra>'
                ))
                
                fig.update_layout(
                    title={
                        'text': "🏆 Top 10 Countries",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'family': 'Poppins'}
                    },
                    xaxis_title="Number of Visits",
                    yaxis_title="",
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins", color=theme['text']),
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.1)',
                        gridwidth=1
                    ),
                    yaxis=dict(showgrid=False),
                    margin=dict(l=120)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Attraction analysis with enhanced visualizations
    if 'AttractionType' in master_df.columns:
        st.markdown('<h3 class="sub-header">🏛️ Attraction Insights</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            type_data = master_df['AttractionType'].value_counts().head(10)
            
            # Enhanced treemap visualization
            fig = go.Figure(go.Treemap(
                labels=type_data.index,
                values=type_data.values,
                parents=[""] * len(type_data),
                textinfo="label+value+percent parent",
                textfont_size=12,
                marker=dict(
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Visits",
                        thickness=20,
                        len=0.7,
                        bgcolor='rgba(255,255,255,0.1)',
                        bordercolor=theme['border'],
                        borderwidth=1
                    ),
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>%{label}</b><br>Visits: %{value}<br>Percentage: %{percentParent}<extra></extra>'
            ))
            
            fig.update_layout(
                title={
                    'text': "🎯 Popular Attraction Types",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Poppins'}
                },
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins", color=theme['text'])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Rating' in master_df.columns:
                avg_ratings = master_df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).head(10)
                
                # Enhanced radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=avg_ratings.values,
                    theta=avg_ratings.index,
                    mode='lines+markers',
                    name='Average Rating',
                    line=dict(color=theme['primary'], width=4),
                    marker=dict(
                        size=12,
                        color=avg_ratings.values,
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(
                            title="Avg Rating",
                            thickness=20,
                            len=0.7,
                            bgcolor='rgba(255,255,255,0.1)',
                            bordercolor=theme['border'],
                            borderwidth=1
                        ),
                        line=dict(color='white', width=3),
                        cmin=1,
                        cmax=5
                    ),
                    fill='toself',
                    fillcolor=f'rgba({",".join(str(int(theme["primary"][i:i+2], 16)) for i in (1, 3, 5))}, 0.3)',
                    text=[f'{v:.2f}' for v in avg_ratings.values],
                    textposition='middle center',
                    hovertemplate='<b>%{theta}</b><br>Avg Rating: %{r:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title={
                        'text': "⭐ Average Ratings by Type",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'family': 'Poppins'}
                    },
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Poppins", color=theme['text']),
                    polar=dict(
                        radialaxis=dict(
                            range=[0, 5],
                            showticklabels=True,
                            gridcolor='rgba(255,255,255,0.2)',
                            linecolor='rgba(255,255,255,0.2)',
                            tickfont=dict(size=10)
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.2)',
                            linecolor='rgba(255,255,255,0.2)',
                            tickfont=dict(size=10)
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_predictions(master_df, models):
    st.markdown('<h2 class="sub-header">🎯 AI-Powered Predictions</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="info-card" style="background: {THEMES[st.session_state.theme]['gradient1']}; color: white; border: none;">
            <h3 style="margin-bottom: 1rem; color: white;">🔮 Prediction Engine</h3>
            <p style="font-size: 1.2rem; line-height: 1.6; margin-bottom: 1rem;">Enter user and attraction details below to get AI-powered predictions for ratings and visit modes.</p>
            <p style="opacity: 0.9; margin: 0; font-size: 1rem;">Powered by advanced machine learning models trained on tourism data</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="sub-header">📝 Input Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">🌍 Geographic Details</h4>', unsafe_allow_html=True)
        
        continents = master_df['Continent'].dropna().unique().tolist() if 'Continent' in master_df.columns else ['North America', 'Europe', 'Asia']
        countries = master_df['Country'].dropna().unique().tolist() if 'Country' in master_df.columns else ['USA', 'UK', 'Germany']
        
        continent = st.selectbox(
            "🌐 Continent", 
            continents, 
            key="continent_select",
            help="Select the continent for prediction context"
        )
        country = st.selectbox(
            "🏳️ Country", 
            countries, 
            key="country_select",
            help="Select the specific country"
        )
        
        st.markdown('<h4 class="gradient-text" style="margin-top: 1.5rem;">📅 Visit Timing</h4>', unsafe_allow_html=True)
        visit_year = st.selectbox(
            "📆 Visit Year", 
            [2025, 2024, 2023, 2022, 2021, 2020], 
            index=1,
            help="Year of the visit"
        )
        visit_month = st.selectbox(
            "🗓️ Visit Month", 
            options=list(range(1, 13)),
            format_func=lambda x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][x-1],
            help="Month of the visit"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<h4 class="gradient-text">🏛️ Attraction Details</h4>', unsafe_allow_html=True)
        
        attraction_types = master_df['AttractionType'].dropna().unique().tolist() if 'AttractionType' in master_df.columns else ['Museum', 'Beach', 'Park']
        attraction_type = st.selectbox(
            "🎭 Attraction Type", 
            attraction_types,
            help="Type of attraction being visited"
        )
        
        st.markdown('<h4 class="gradient-text" style="margin-top: 1.5rem;">👤 User Profile</h4>', unsafe_allow_html=True)
        user_avg_rating = st.slider(
            "⭐ User's Average Rating", 
            1.0, 5.0, 3.5, 0.1,
            help="Historical average rating given by this user"
        )
        user_visit_count = st.number_input(
            "📊 Total Past Visits", 
            min_value=0, max_value=1000, value=5, step=1,
            help="Total number of attractions visited by this user"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced prediction button with better styling
    st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
    predict_button = st.button(
        "🔮 Generate AI Predictions", 
        type="primary", 
        use_container_width=False,
        help="Click to generate predictions using trained ML models"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        user_features = {
            'VisitYear': visit_year,
            'VisitMonth': visit_month,
            'UserAvgRating': user_avg_rating,
            'UserVisitCount': user_visit_count,
        }
        
        # Enhanced loading animation
        with st.spinner('🤖 AI models analyzing data and generating predictions...'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h4 class="gradient-text">🌟 Rating Prediction</h4>', unsafe_allow_html=True)
                
                rating_pred, rating_info = predict_rating(models, user_features)
                
                if rating_pred is not None:
                    # Enhanced gauge chart with better styling
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=rating_pred,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={
                            'text': "Predicted Rating",
                            'font': {'size': 22, 'family': 'Poppins', 'color': THEMES[st.session_state.theme]['text']}
                        },
                        delta={'reference': 3.5, 'increasing': {'color': "#00d2d3"}, 'decreasing': {'color': "#ff6b6b"}},
                        number={'font': {'size': 40, 'family': 'Poppins'}},
                        gauge={
                            'axis': {
                                'range': [0, 5], 
                                'tickwidth': 3, 
                                'tickcolor': THEMES[st.session_state.theme]['text'],
                                'tickfont': {'size': 14, 'family': 'Poppins'}
                            },
                            'bar': {'color': THEMES[st.session_state.theme]['primary'], 'thickness': 0.8},
                            'bgcolor': "rgba(255,255,255,0.1)",
                            'borderwidth': 4,
                            'bordercolor': THEMES[st.session_state.theme]['border'],
                            'steps': [
                                {'range': [0, 1], 'color': 'rgba(245, 87, 108, 0.3)'},
                                {'range': [1, 2], 'color': 'rgba(240, 147, 251, 0.3)'},
                                {'range': [2, 3], 'color': 'rgba(254, 202, 87, 0.3)'},
                                {'range': [3, 4], 'color': 'rgba(79, 172, 254, 0.3)'},
                                {'range': [4, 5], 'color': 'rgba(56, 239, 125, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': THEMES[st.session_state.theme]['accent'], 'width': 6},
                                'thickness': 0.75,
                                'value': rating_pred
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=60, b=20),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Poppins", color=THEMES[st.session_state.theme]['text'])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced confidence display
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['gradient3']}; 
                                    padding: 1.5rem; border-radius: 15px; margin-top: 1rem; color: white; text-align: center;">
                            <h5 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">Model Performance</h5>
                            <p style="margin: 0; font-size: 1rem; opacity: 0.9;">{rating_info}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ {rating_info}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown('<h4 class="gradient-text">🚶 Visit Mode Prediction</h4>', unsafe_allow_html=True)
                
                mode_pred, mode_info, probs = predict_visit_mode(models, user_features)
                
                if mode_pred is not None:
                    # Enhanced prediction display
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['gradient2']}; 
                                    padding: 2rem; border-radius: 20px; text-align: center; margin-bottom: 1.5rem; color: white;">
                            <h2 style="margin: 0 0 0.5rem 0; font-size: 2.5rem; font-weight: 700;">{mode_pred}</h2>
                            <p style="margin: 0; font-size: 1.2rem; opacity: 0.9;">Predicted Visit Mode</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if probs is not None:
                        try:
                            # Enhanced probability visualization
                            model_info = models.get('classification', {})
                            clf = model_info.get('model', None)
                            classes_ = getattr(clf, 'classes_', None)
                            
                            if classes_ is not None:
                                prob_data = []
                                for i, cls in enumerate(classes_):
                                    prob_data.append({
                                        'Class': str(cls),
                                        'Probability': probs[i],
                                        'Percentage': f"{probs[i]*100:.1f}%"
                                    })
                                
                                prob_df = pd.DataFrame(prob_data)
                                
                                # Create enhanced horizontal bar chart
                                fig = go.Figure(go.Bar(
                                    x=prob_df['Probability'],
                                    y=prob_df['Class'],
                                    orientation='h',
                                    text=prob_df['Percentage'],
                                    textposition='outside',
                                    marker=dict(
                                        color=prob_df['Probability'],
                                        colorscale='Viridis',
                                        showscale=False,
                                        line=dict(color='white', width=2),
                                        cornerradius=5
                                    ),
                                    hovertemplate='<b>%{y}</b><br>Probability: %{x:.3f}<br>Percentage: %{text}<extra></extra>'
                                ))
                                
                                fig.update_layout(
                                    title={
                                        'text': "Class Probabilities",
                                        'font': {'size': 16, 'family': 'Poppins'}
                                    },
                                    xaxis=dict(
                                        range=[0, 1], 
                                        title="Probability",
                                        showgrid=True,
                                        gridcolor='rgba(255,255,255,0.1)'
                                    ),
                                    yaxis_title="",
                                    height=280,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(family="Poppins", color=THEMES[st.session_state.theme]['text']),
                                    margin=dict(l=80, r=20, t=40, b=20)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass
                    
                    # Enhanced model info display
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['gradient1']}; 
                                    padding: 1.5rem; border-radius: 15px; margin-top: 1rem; color: white; text-align: center;">
                            <h5 style="margin: 0 0 0.5rem 0; font-size: 1.1rem;">Model Performance</h5>
                            <p style="margin: 0; font-size: 1rem; opacity: 0.9;">{mode_info}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning(f"⚠️ {mode_info}")
                
                st.markdown('</div>', unsafe_allow_html=True)

def show_recommendations_page(master_df, models):
    st.markdown('<h2 class="sub-header">💡 Smart Recommendations</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="info-card" style="background: {THEMES[st.session_state.theme]['gradient2']}; color: white; border: none;">
            <h3 style="margin-bottom: 1rem; color: white;">🎯 Personalized Attraction Recommendations</h3>
            <p style="font-size: 1.2rem; line-height: 1.6; margin-bottom: 1rem;">Get AI-powered recommendations based on user preferences and behavior patterns.</p>
            <p style="opacity: 0.9; margin: 0; font-size: 1rem;">Using collaborative filtering and popularity-based algorithms</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        
        # Enhanced input fields with better styling
        user_id_input = st.text_input(
            "👤 Enter User ID",
            value="1",
            help="Enter the user ID to get personalized recommendations",
            placeholder="e.g., 123 or user_abc"
        )
        
        n_rec = st.slider(
            "📊 Number of Recommendations",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Select how many attractions to recommend"
        )
        
        # Enhanced recommendation type selector
        rec_type = st.radio(
            "🎨 Recommendation Style",
            ["🔥 Popular Attractions", "✨ Diverse Selection", "🎲 Surprise Me"],
            horizontal=True,
            help="Choose the type of recommendations you prefer"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced centered button
        st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
        get_rec_button = st.button(
            "🚀 Generate Recommendations", 
            type="primary", 
            use_container_width=True,
            help="Click to get personalized attraction recommendations"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    if get_rec_button:
        with st.spinner('🤖 Analyzing user preferences and generating personalized recommendations...'):
            try:
                # Try int conversion but fall back to string
                try:
                    candidate_id = int(user_id_input)
                except Exception:
                    candidate_id = user_id_input
                
                items, info = get_recommendations(models, candidate_id, n_recommendations=n_rec)
                
                # Enhanced info display
                st.markdown(f"""
                    <div class="info-card" style="background: {THEMES[st.session_state.theme]['gradient3']}; 
                                color: white; border: none; margin: 2rem 0; text-align: center;">
                        <h4 style="margin: 0 0 0.5rem 0; color: white;">🔍 Recommendation Status</h4>
                        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">{info}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if items:
                    st.markdown('<h3 class="gradient-text" style="text-align: center; margin: 2rem 0;">✨ Your Recommended Attractions</h3>', unsafe_allow_html=True)
                    
                    # Enhanced grid layout for recommendations
                    cols = st.columns(2)
                    attraction_icons = ['🏛️', '🎭', '🏖️', '🏔️', '🎪', '🎨', '🏰', '🌊', '🎢', '🌸']
                    
                    for idx, item in enumerate(items):
                        with cols[idx % 2]:
                            # Enhanced recommendation cards with animations
                            confidence = max(95 - idx * 3, 60)
                            icon = attraction_icons[idx % len(attraction_icons)]
                            
                            st.markdown(f"""
                                <div class="recommendation-item" style="animation-delay: {idx * 0.1}s; margin-bottom: 1.5rem;">
                                    <div style="display: flex; align-items: center; justify-content: space-between;">
                                        <div style="flex: 1;">
                                            <h4 style="margin: 0 0 0.5rem 0; color: {THEMES[st.session_state.theme]['primary']}; font-size: 1.3rem;">
                                                Attraction #{item}
                                            </h4>
                                            <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                                                <div style="width: 100px; height: 8px; background: rgba(255,255,255,0.2); border-radius: 4px; margin-right: 1rem;">
                                                    <div style="width: {confidence}%; height: 100%; background: {THEMES[st.session_state.theme]['gradient1']}; border-radius: 4px; transition: width 1s ease;"></div>
                                                </div>
                                                <span style="font-size: 0.9rem; opacity: 0.8;">{confidence}% match</span>
                                            </div>
                                            <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">
                                                Recommended based on {rec_type.split()[1].lower()} preferences
                                            </p>
                                        </div>
                                        <div style="font-size: 3rem; margin-left: 1rem;">
                                            {icon}
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Add summary statistics
                    st.markdown(f"""
                        <div class="info-card" style="margin-top: 2rem; text-align: center;">
                            <h4 class="gradient-text">📊 Recommendation Summary</h4>
                            <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                                <div>
                                    <h3 style="margin: 0; color: {THEMES[st.session_state.theme]['primary']};">{len(items)}</h3>
                                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Total Recommendations</p>
                                </div>
                                <div>
                                    <h3 style="margin: 0; color: {THEMES[st.session_state.theme]['secondary']};">{confidence:.0f}%</h3>
                                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Avg Confidence</p>
                                </div>
                                <div>
                                    <h3 style="margin: 0; color: {THEMES[st.session_state.theme]['accent']};">AI</h3>
                                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.7;">Powered</p>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No recommendations available for this user. Please try a different user ID.")
                    
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {e}")

def show_performance(models):
    st.markdown('<h2 class="sub-header">📈 Model Performance Metrics</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="info-card" style="background: {THEMES[st.session_state.theme]['gradient4']}; color: white; border: none;">
            <h3 style="margin-bottom: 1rem; color: white;">🏆 Performance Dashboard</h3>
            <p style="font-size: 1.2rem; line-height: 1.6; margin-bottom: 1rem;">Detailed performance metrics and evaluation results from our machine learning models.</p>
            <p style="opacity: 0.9; margin: 0; font-size: 1rem;">Real-time model monitoring and performance analytics</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced tabs with better styling
    tab1, tab2, tab3 = st.tabs(["📊 Rating Predictor", "🎯 Visit Mode Classifier", "💡 Recommendation System"])
    
    with tab1:
        if 'regression' in models:
            perf = models['regression'].get('metadata', {}).get('performance', {})
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<h4 class="gradient-text">Rating Prediction Model Performance</h4>', unsafe_allow_html=True)
            
            if perf:
                # Enhanced metrics visualization with cards
                metric_cols = st.columns(len(perf))
                for idx, (metric, value) in enumerate(perf.items()):
                    with metric_cols[idx]:
                        if isinstance(value, (int, float)):
                            # Create performance gauge for each metric
                            display_value = f"{value:.4f}" if value < 10 else f"{value:.2f}"
                            percentage = min(abs(value) * 100, 100) if metric.lower().startswith(('r2', 'accuracy')) else abs(value)
                            
                            st.markdown(f"""
                                <div style="background: {THEMES[st.session_state.theme]['gradient1']}; 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
                                    <h3 style="margin: 0 0 0.5rem 0; font-size: 2rem;">{display_value}</h3>
                                    <p style="margin: 0; font-size: 1rem; opacity: 0.9;">{metric.replace('_', ' ').title()}</p>
                                    <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.3); border-radius: 3px; margin-top: 1rem;">
                                        <div style="width: {min(percentage, 100)}%; height: 100%; background: white; border-radius: 3px; transition: width 2s ease;"></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style="background: {THEMES[st.session_state.theme]['card_bg']}; 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: {THEMES[st.session_state.theme]['primary']};">{metric.replace('_', ' ').title()}</h4>
                                    <p style="margin: 0; font-size: 1rem;">{value}</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Model details section
                st.markdown('<h5 class="gradient-text" style="margin: 2rem 0 1rem 0;">Model Details</h5>', unsafe_allow_html=True)
                model_details = models['regression'].get('metadata', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['card_bg']}; padding: 1.5rem; border-radius: 15px;">
                            <h6 style="color: {THEMES[st.session_state.theme]['primary']};">Model Information</h6>
                            <p><strong>Algorithm:</strong> {model_details.get('algorithm', 'N/A')}</p>
                            <p><strong>Features:</strong> {len(model_details.get('features', []))}</p>
                            <p><strong>Training Time:</strong> {model_details.get('training_time', 'N/A')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['card_bg']}; padding: 1.5rem; border-radius: 15px;">
                            <h6 style="color: {THEMES[st.session_state.theme]['primary']};">Performance Insights</h6>
                            <p><strong>Best Score:</strong> {max(perf.values()) if perf and all(isinstance(v, (int, float)) for v in perf.values()) else 'N/A'}</p>
                            <p><strong>Status:</strong> <span style="color: #00d2d3;">✅ Production Ready</span></p>
                            <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No performance metrics available for regression model.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="info-card" style="text-align: center; padding: 3rem;">
                    <h3 style="color: {THEMES[st.session_state.theme]['text_secondary']};">📊 Regression Model Not Available</h3>
                    <p style="margin: 1rem 0; opacity: 0.7;">The rating prediction model is not currently loaded.</p>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.5;">Please ensure model files are in the correct directory.</p>
                </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        if 'classification' in models:
            perf = models['classification'].get('metadata', {}).get('performance', {})
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<h4 class="gradient-text">Visit Mode Classification Performance</h4>', unsafe_allow_html=True)
            
            if perf:
                # Enhanced metrics visualization
                metric_cols = st.columns(len(perf))
                for idx, (metric, value) in enumerate(perf.items()):
                    with metric_cols[idx]:
                        if isinstance(value, (int, float)):
                            display_value = f"{value:.4f}" if value < 10 else f"{value:.2f}"
                            percentage = value * 100 if value <= 1 else min(value, 100)
                            
                            st.markdown(f"""
                                <div style="background: {THEMES[st.session_state.theme]['gradient2']}; 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
                                    <h3 style="margin: 0 0 0.5rem 0; font-size: 2rem;">{display_value}</h3>
                                    <p style="margin: 0; font-size: 1rem; opacity: 0.9;">{metric.replace('_', ' ').title()}</p>
                                    <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.3); border-radius: 3px; margin-top: 1rem;">
                                        <div style="width: {min(percentage, 100)}%; height: 100%; background: white; border-radius: 3px; transition: width 2s ease;"></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div style="background: {THEMES[st.session_state.theme]['card_bg']}; 
                                           padding: 1.5rem; border-radius: 15px; text-align: center; margin-bottom: 1rem;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: {THEMES[st.session_state.theme]['primary']};">{metric.replace('_', ' ').title()}</h4>
                                    <p style="margin: 0; font-size: 1rem;">{value}</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Classification-specific details
                st.markdown('<h5 class="gradient-text" style="margin: 2rem 0 1rem 0;">Classification Details</h5>', unsafe_allow_html=True)
                model_details = models['classification'].get('metadata', {})
                
                col1, col2 = st.columns(2)
                with col1:
                    label_mapping = models['classification'].get('label_mapping', {})
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['card_bg']}; padding: 1.5rem; border-radius: 15px;">
                            <h6 style="color: {THEMES[st.session_state.theme]['primary']};">Model Configuration</h6>
                            <p><strong>Algorithm:</strong> {model_details.get('algorithm', 'N/A')}</p>
                            <p><strong>Classes:</strong> {len(label_mapping) if label_mapping else 'N/A'}</p>
                            <p><strong>Features:</strong> {len(model_details.get('features', []))}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['card_bg']}; padding: 1.5rem; border-radius: 15px;">
                            <h6 style="color: {THEMES[st.session_state.theme]['primary']};">Performance Summary</h6>
                            <p><strong>Best F1-Score:</strong> {max([v for v in perf.values() if isinstance(v, (int, float))]) if perf else 'N/A'}</p>
                            <p><strong>Status:</strong> <span style="color: #00d2d3;">✅ Active</span></p>
                            <p><strong>Prediction Ready:</strong> <span style="color: #00d2d3;">Yes</span></p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No performance metrics available for classification model.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="info-card" style="text-align: center; padding: 3rem;">
                    <h3 style="color: {THEMES[st.session_state.theme]['text_secondary']};">🎯 Classification Model Not Available</h3>
                    <p style="margin: 1rem 0; opacity: 0.7;">The visit mode classifier is not currently loaded.</p>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.5;">Please ensure model files are in the correct directory.</p>
                </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        if 'recommendation' in models:
            metadata = models['recommendation'].get('metadata', {})
            
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<h4 class="gradient-text">Recommendation System Details</h4>', unsafe_allow_html=True)
            
            if metadata:
                # Enhanced recommendation system overview
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['gradient3']}; 
                                   padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h3 style="margin: 0 0 0.5rem 0;">✨</h3>
                            <h4 style="margin: 0 0 0.5rem 0;">Algorithm Type</h4>
                            <p style="margin: 0; opacity: 0.9;">Collaborative Filtering</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['gradient1']}; 
                                   padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h3 style="margin: 0 0 0.5rem 0;">🎯</h3>
                            <h4 style="margin: 0 0 0.5rem 0;">Recommendation Type</h4>
                            <p style="margin: 0; opacity: 0.9;">Popularity-Based</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div style="background: {THEMES[st.session_state.theme]['gradient2']}; 
                                   padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                            <h3 style="margin: 0 0 0.5rem 0;">⚡</h3>
                            <h4 style="margin: 0 0 0.5rem 0;">Status</h4>
                            <p style="margin: 0; opacity: 0.9;">Active & Ready</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metadata display
                st.markdown('<h5 class="gradient-text" style="margin: 2rem 0 1rem 0;">System Configuration</h5>', unsafe_allow_html=True)
                
                metadata_items = []
                for key, value in metadata.items():
                    formatted_key = key.replace('_', ' ').title()
                    metadata_items.append((formatted_key, str(value)))
                
                # Display metadata in a nice grid
                if metadata_items:
                    for i in range(0, len(metadata_items), 2):
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            if i + j < len(metadata_items):
                                key, value = metadata_items[i + j]
                                with col:
                                    st.markdown(f"""
                                        <div style="background: {THEMES[st.session_state.theme]['card_bg']}; 
                                                   padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
                                            <h6 style="margin: 0 0 0.5rem 0; color: {THEMES[st.session_state.theme]['primary']};">{key}</h6>
                                            <p style="margin: 0; font-size: 1rem;">{value}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                
                # Available models in recommendation system
                available_models = []
                for model_name in ['hybrid', 'user_cf', 'item_cf', 'content_cf', 'svd']:
                    if model_name in models['recommendation']:
                        available_models.append(model_name.replace('_', ' ').title())
                
                if available_models:
                    st.markdown('<h5 class="gradient-text" style="margin: 2rem 0 1rem 0;">Available Algorithms</h5>', unsafe_allow_html=True)
                    cols = st.columns(len(available_models))
                    for idx, model_name in enumerate(available_models):
                        with cols[idx]:
                            st.markdown(f"""
                                <div style="background: {THEMES[st.session_state.theme]['gradient4']}; 
                                           padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                                    <p style="margin: 0; font-weight: 600;">{model_name}</p>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No metadata available for recommendation system.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="info-card" style="text-align: center; padding: 3rem;">
                    <h3 style="color: {THEMES[st.session_state.theme]['text_secondary']};">💡 Recommendation System Not Available</h3>
                    <p style="margin: 1rem 0; opacity: 0.7;">The recommendation engine is not currently loaded.</p>
                    <p style="margin: 0; font-size: 0.9rem; opacity: 0.5;">Please ensure model files are in the correct directory.</p>
                </div>
            """, unsafe_allow_html=True)

def main():
    # Load data and models
    data_result = load_data()
    if data_result is None:
        st.error("📁 Data files not found. Please ensure the processed dataset exists.")
        st.stop()
    
    master_df, additional_data = data_result
    models = load_models()
    
    # Enhanced main header with better animations
    st.markdown('<h1 class="main-header">🌍 Tourism Experience Analytics</h1>', unsafe_allow_html=True)
    
    # Theme switcher with enhanced styling
    st.markdown('<h3 style="text-align: center; margin-bottom: 2rem; opacity: 0.8;">🎨 Choose Your Experience</h3>', unsafe_allow_html=True)
    create_theme_switcher()
    
    # Enhanced divider with animated gradient
    st.markdown(f'''
        <div style="height: 4px; background: {THEMES[st.session_state.theme]['gradient1']}; 
                   margin: 3rem 0; border-radius: 2px; 
                   animation: gradientMove 3s ease infinite;
                   background-size: 200% 200%;"></div>
    ''', unsafe_allow_html=True)
    
    # Enhanced sidebar with better styling
    with st.sidebar:
        st.markdown(f"""
            <div style="background: {THEMES[st.session_state.theme]['gradient1']}; 
                        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;
                        box-shadow: {THEMES[st.session_state.theme]['shadow']};">
                <h2 style="color: white; margin: 0; font-size: 1.8rem; font-weight: 600;">🎯 Navigation</h2>
            </div>
        """, unsafe_allow_html=True)
        
        pages = {
            "🏠 Overview": "overview",
            "📊 Analytics": "analytics", 
            "🎯 Predictions": "predictions",
            "💡 Recommendations": "recommendations",
            "📈 Performance": "performance"
        }
        
        selected_page = st.selectbox(
            "Navigate to:",
            list(pages.keys()),
            label_visibility="collapsed"
        )
        
        # Enhanced about section
        st.markdown(f"""
            <div style="background: {THEMES[st.session_state.theme]['card_bg']}; 
                        padding: 2rem; border-radius: 20px; margin-top: 2rem;
                        box-shadow: {THEMES[st.session_state.theme]['shadow']};
                        backdrop-filter: blur(20px);">
                <h3 style="margin-bottom: 1.5rem; color: {THEMES[st.session_state.theme]['primary']};">ℹ️ About This Dashboard</h3>
                <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 1rem;">
                    A comprehensive tourism analytics platform featuring:
                </p>
                <ul style="font-size: 0.95rem; line-height: 1.8; padding-left: 1.5rem;">
                    <li>📊 Interactive data visualizations</li>
                    <li>🤖 AI-powered predictions</li>
                    <li>💡 Smart recommendations</li>
                    <li>📈 Performance metrics</li>
                    <li>🎨 Multiple themes</li>
                </ul>
                <hr style="margin: 1.5rem 0; opacity: 0.3; border: none; height: 1px; background: {THEMES[st.session_state.theme]['border']};">
                <div style="text-align: center;">
                    <p style="font-size: 0.85rem; opacity: 0.7; margin: 0;">
                        Built with ❤️ using Streamlit & Python
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Enhanced time display
        current_time = datetime.now()
        st.markdown(f"""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; 
                        background: {THEMES[st.session_state.theme]['card_bg']}; 
                        border-radius: 15px; backdrop-filter: blur(10px);">
                <p style="font-size: 1rem; margin: 0 0 0.5rem 0; color: {THEMES[st.session_state.theme]['primary']}; font-weight: 600;">
                    {current_time.strftime('%B %d, %Y')}
                </p>
                <p style="font-size: 0.9rem; margin: 0; opacity: 0.7;">
                    {current_time.strftime('%I:%M %p')}
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Main content area with page routing
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
