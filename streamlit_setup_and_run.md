# Tourism Analytics Streamlit Application - Setup & Run Guide

## Directory Structure Setup

Your project should have this structure:

```
tourism-analytics/
├── data/
│   ├── raw/                    # Original Excel files
│   └── processed/              # Processed datasets from Step 2
│       ├── master_dataset.csv
│       ├── user_item_matrix.csv
│       └── recommendation_data.csv
├── models/
│   ├── regression/             # From Step 3
│   │   ├── best_rating_predictor.pkl
│   │   ├── rating_predictor_scaler.pkl
│   │   └── rating_predictor_metadata.json
│   ├── classification/         # From Step 4
│   │   ├── best_visitmode_classifier.pkl
│   │   ├── visitmode_classifier_scaler.pkl
│   │   ├── visitmode_label_mapping.pkl
│   │   └── visitmode_classifier_metadata.json
│   └── recommendation/         # From Step 5
│       ├── hybrid_recommender.pkl
│       ├── user_based_cf.pkl
│       ├── item_based_cf.pkl
│       ├── content_based_cf.pkl
│       ├── svd_recommender.pkl
│       ├── training_matrix.csv
│       └── recommendation_metadata.json
├── notebooks/                  # Your Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning_preprocessing.ipynb
│   ├── 03_regression_modeling.ipynb
│   ├── 04_classification_modeling.ipynb
│   └── 05_recommendation_system.ipynb
├── streamlit_app/
│   └── app.py                  # Main Streamlit application
├── requirements_streamlit.txt
└── README.md
```

## Setup Instructions

### 1. Create Streamlit Directory

```bash
# From your project root directory
mkdir streamlit_app
```

### 2. Copy Application File

Save the `app.py` file from the artifact into the `streamlit_app/` directory.

### 3. Install Streamlit Dependencies

```bash
# Activate your virtual environment first
tourism_env\Scripts\activate  # Windows
# source tourism_env/bin/activate  # macOS/Linux

# Install Streamlit requirements
pip install -r requirements_streamlit.txt
```

### 4. Verify Model Files

Ensure you have completed Steps 2-5 and have these files:

**Required Files:**
- `data/processed/master_dataset.csv`
- At least one model from each category (regression, classification, recommendation)

**Check Model Availability:**

```python
import os

# Check data
print("Data files:")
print(f"✓ Master dataset: {os.path.exists('data/processed/master_dataset.csv')}")

# Check models
print("\nModel files:")
print(f"✓ Regression: {os.path.exists('models/regression/best_rating_predictor.pkl')}")
print(f"✓ Classification: {os.path.exists('models/classification/best_visitmode_classifier.pkl')}")
print(f"✓ Recommendation: {os.path.exists('models/recommendation/hybrid_recommender.pkl')}")
```

## Running the Application

### Option 1: From Project Root

```bash
# From tourism-analytics/ directory
streamlit run streamlit_app/app.py
```

### Option 2: From Streamlit Directory

```bash
# Navigate to streamlit_app directory
cd streamlit_app

# Run the application
streamlit run app.py
```

## Application Features

### 📊 Overview Page
- Project statistics and key metrics
- Model status indicators
- Dataset information summary
- Rating distribution visualization

### 📈 Data Analytics
- Temporal analysis (yearly/monthly trends)
- Geographic distribution analysis
- Attraction type popularity
- Visit mode patterns

### 🎯 Predictions
- **Rating Prediction**: Enter user and attraction details to predict satisfaction ratings
- **Visit Mode Classification**: Predict whether users will travel for business, family, couples, friends, or solo
- Interactive input forms with real-time predictions

### 💡 Recommendations
- **Personalized Recommendations**: Select a user and get tailored attraction suggestions
- Uses the best-performing recommendation algorithm from Step 5
- Shows user history and recommendation explanations

### 📈 Model Performance
- Detailed performance metrics for all three models
- Model comparison dashboard
- Training statistics and metadata

## Troubleshooting

### Common Issues

**1. "Data files not found"**
- Ensure you've completed Step 2 (Data Preprocessing)
- Check that `data/processed/master_dataset.csv` exists

**2. "Models not available"**
- Complete Steps 3-5 to train all models
- Verify model files exist in `models/` subdirectories

**3. "Module not found" errors**
- Install missing packages: `pip install package_name`
- Ensure virtual environment is activated

**4. Application won't start**
- Check port availability (default: 8501)
- Run: `streamlit run app.py --server.port 8502` for different port

### Performance Optimization

**For Large Datasets:**
- Consider sampling data for faster loading
- Use `@st.cache_data` and `@st.cache_resource` decorators (already implemented)

**Memory Issues:**
- Reduce model complexity if needed
- Consider loading models on-demand rather than at startup

## Deployment Options

### Local Development
```bash
streamlit run streamlit_app/app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy with one click

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py"]
```

## Application URLs

Once running, access:
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501 (if accessible from other devices)

The application will automatically open in your default web browser.

## Features Summary

✅ **Complete ML Pipeline Integration**
- Rating prediction with confidence scores
- Visit mode classification with probabilities  
- Personalized attraction recommendations

✅ **Interactive Dashboard**
- Responsive design with modern UI
- Real-time predictions and visualizations
- User-friendly input forms

✅ **Business Intelligence**
- Data analytics and trend analysis
- Model performance monitoring
- Actionable insights for tourism operators

This Streamlit application provides a complete, production-ready interface for your Tourism Analytics system!