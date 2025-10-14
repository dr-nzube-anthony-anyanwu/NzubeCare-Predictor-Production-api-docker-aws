import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NzubeCare - Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
import os
try:
    # First try environment variable (Docker)
    API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
    # If running in Docker and no explicit URL, use internal service name
    if API_BASE_URL == 'http://localhost:8000' and os.getenv('ENVIRONMENT') == 'development':
        API_BASE_URL = 'http://api:8000'
except:
    API_BASE_URL = 'http://localhost:8000'  # Default fallback

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; padding-bottom: 10px;}
    .section-header {font-size: 2rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px;}
    .feature-importance {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .prediction-high {color: #ff4b4b; font-weight: bold; font-size: 1.5rem;}
    .prediction-medium {color: #ffa14b; font-weight: bold; font-size: 1.5rem;}
    .prediction-low {color: #00cc96; font-weight: bold; font-size: 1.5rem;}
    .api-status-healthy {background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 10px; margin: 10px 0;}
    .api-status-unhealthy {background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; padding: 10px; margin: 10px 0;}
    .warning-box {background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🏥 NzubeCare Readmission Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predictive Analytics for Hospital Readmission Risk (API-Powered)")
st.markdown("---")

# Check API health
@st.cache_data(ttl=60)  # Cache for 1 minute
def check_api_health():
    """Check if the API is healthy and accessible."""
    try:
        # Try both health endpoints for compatibility
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        # If load balancer routes to /api, try that endpoint too
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API returned status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def predict_via_api(patient_data):
    """Make prediction via API."""
    try:
        # First try the new /api/predict endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"API error: {response.status_code} - {response.text}"}
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

# API Status Check
api_healthy, api_status = check_api_health()

if api_healthy:
    st.markdown(f'''
    <div class="api-status-healthy">
        ✅ <strong>API Status:</strong> Healthy | Model Loaded: {api_status.get("model_loaded", "Unknown")}
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown(f'''
    <div class="api-status-unhealthy">
        ❌ <strong>API Status:</strong> Unhealthy | Error: {api_status.get("error", "Unknown error")}
    </div>
    ''', unsafe_allow_html=True)
    st.warning("⚠️ The prediction API is not available. Please check if the API server is running.")
    st.info(f"Expected API URL: {API_BASE_URL}")
    st.stop()

# Sidebar for patient information
st.sidebar.markdown('<h2 class="section-header">Patient Information</h2>', unsafe_allow_html=True)

# Patient Demographics
st.sidebar.markdown("### Demographics")
race = st.sidebar.selectbox(
    "Race", 
    ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"],
    help="Patient's racial background"
)

gender = st.sidebar.selectbox(
    "Gender", 
    ["Female", "Male"],
    help="Patient's biological gender"
)

age_num = st.sidebar.number_input(
    "Age", 
    min_value=5, 
    max_value=95, 
    value=65,
    help="Patient's age in years"
)

# Hospitalization Details
st.sidebar.markdown("### Hospitalization Details")
time_in_hospital = st.sidebar.number_input(
    "Days in Hospital", 
    min_value=1, 
    max_value=14, 
    value=3,
    help="Number of days patient stayed in hospital"
)

num_lab_procedures = st.sidebar.number_input(
    "Lab Procedures", 
    min_value=1, 
    max_value=100, 
    value=40,
    help="Number of laboratory procedures performed"
)

num_procedures = st.sidebar.number_input(
    "Medical Procedures", 
    min_value=0, 
    max_value=10, 
    value=1,
    help="Number of medical procedures performed"
)

num_medications = st.sidebar.number_input(
    "Medications", 
    min_value=1, 
    max_value=30, 
    value=10,
    help="Number of distinct medications administered"
)

number_outpatient = st.sidebar.number_input(
    "Outpatient Visits (Past Year)", 
    min_value=0, 
    max_value=20, 
    value=0,
    help="Number of outpatient visits in the past year"
)

number_emergency = st.sidebar.number_input(
    "Emergency Visits (Past Year)", 
    min_value=0, 
    max_value=20, 
    value=0,
    help="Number of emergency visits in the past year"
)

number_inpatient = st.sidebar.number_input(
    "Inpatient Visits (Past Year)", 
    min_value=0, 
    max_value=10, 
    value=0,
    help="Number of inpatient visits in the past year"
)

number_diagnoses = st.sidebar.number_input(
    "Number of Diagnoses", 
    min_value=1, 
    max_value=20, 
    value=5,
    help="Number of diagnoses recorded"
)

# Medical Test Results
st.sidebar.markdown("### Medical Test Results")
max_glu_serum = st.sidebar.selectbox(
    "Max Glucose Serum", 
    ["Norm", ">200", ">300", "None"],
    help="Maximum glucose serum level during stay"
)

A1Cresult = st.sidebar.selectbox(
    "A1C Test Result", 
    ["Norm", ">7", ">8", "None"],
    help="Hemoglobin A1C test result"
)

# Medication Information
st.sidebar.markdown("### Medication Information")
change = st.sidebar.selectbox(
    "Medication Change", 
    ["No", "Ch"],
    help="Was there a change in diabetic medications?"
)

diabetesMed = st.sidebar.selectbox(
    "Diabetes Medication", 
    ["No", "Yes"],
    help="Was patient prescribed diabetes medication?"
)

# Prepare patient data
patient_data = {
    "race": race,
    "gender": gender,
    "age_num": age_num,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "number_outpatient": number_outpatient,
    "number_emergency": number_emergency,
    "number_inpatient": number_inpatient,
    "number_diagnoses": number_diagnoses,
    "max_glu_serum": max_glu_serum,
    "A1Cresult": A1Cresult,
    "change": change,
    "diabetesMed": diabetesMed
}

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h2 class="section-header">📊 Patient Summary</h2>', unsafe_allow_html=True)
    
    # Patient summary
    summary_df = pd.DataFrame([
        ["Age", f"{age_num} years"],
        ["Gender", gender],
        ["Race", race],
        ["Hospital Stay", f"{time_in_hospital} days"],
        ["Lab Procedures", num_lab_procedures],
        ["Medications", num_medications],
        ["Previous Inpatient Visits", number_inpatient],
        ["Diabetes Medication", diabetesMed]
    ], columns=["Attribute", "Value"])
    
    st.dataframe(summary_df, use_container_width=True)

with col2:
    st.markdown('<h2 class="section-header">🎯 Risk Assessment</h2>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
        with st.spinner("Making prediction via API..."):
            success, result = predict_via_api(patient_data)
            
            if success:
                # Store results in session state
                st.session_state['prediction_result'] = result
                st.success("✅ Prediction completed!")
            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

# Display prediction results
if 'prediction_result' in st.session_state:
    result = st.session_state['prediction_result']
    
    st.markdown('<h2 class="section-header">📈 Prediction Results</h2>', unsafe_allow_html=True)
    
    # Main prediction display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_level = result['risk_level']
        risk_prob = result['risk_probability']
        
        # Color coding based on risk level
        if risk_level == "High":
            risk_class = "prediction-high"
            color = "#ff4b4b"
        elif risk_level == "Medium":
            risk_class = "prediction-medium"
            color = "#ffa14b"
        else:
            risk_class = "prediction-low"
            color = "#00cc96"
        
        st.markdown(f'<p class="{risk_class}">Risk Level: {risk_level}</p>', unsafe_allow_html=True)
        st.metric("Readmission Probability", f"{risk_prob:.1%}")
        
    with col2:
        # Risk gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col3:
        st.metric("Model Version", result.get('model_version', '1.0'))
        st.metric("API Response Time", "< 1s")
    
    # Recommendations
    st.markdown('<h3 class="section-header">💡 Clinical Recommendations</h3>', unsafe_allow_html=True)
    recommendations = result.get('recommendations', [])
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    # Feature Importance
    if result.get('feature_importance') and isinstance(result['feature_importance'], dict):
        st.markdown('<h3 class="section-header">🔍 Key Factors Influencing Prediction</h3>', unsafe_allow_html=True)
        
        feature_importance = result['feature_importance']
        
        # Filter out non-numeric importance values
        numeric_features = {k: v for k, v in feature_importance.items() 
                          if isinstance(v, (int, float))}
        
        if numeric_features:
            # Create feature importance chart
            features = list(numeric_features.keys())[:10]  # Top 10
            importances = list(numeric_features.values())[:10]
            
            fig_importance = px.bar(
                x=importances,
                y=features,
                orientation='h',
                title="Top Contributing Factors",
                labels={'x': 'Importance Score', 'y': 'Features'},
                color=importances,
                color_continuous_scale='blues'
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance data not available for this prediction.")

# Additional Information
st.markdown('<h2 class="section-header">ℹ️ About This Tool</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Model Information:**
    - **Algorithm:** LightGBM Classifier
    - **Purpose:** Predict 30-day hospital readmission risk
    - **Target Population:** Diabetic patients
    - **Deployment:** API-based microservice architecture
    """)

with col2:
    st.markdown("""
    **Risk Categories:**
    - **Low (< 40%):** Standard discharge planning
    - **Medium (40-70%):** Enhanced monitoring recommended
    - **High (≥ 70%):** Intensive care coordination needed
    """)

# API Information
with st.expander("🔧 API Information"):
    st.markdown(f"""
    **API Base URL:** `{API_BASE_URL}`
    
    **Available Endpoints:**
    - `GET /health` - Check API health status
    - `POST /predict` - Single patient prediction
    - `POST /predict/batch` - Batch predictions
    - `GET /model/info` - Model information
    - `GET /docs` - Interactive API documentation
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🏥 NzubeCare Readmission Predictor v2.0 | Powered by FastAPI & Streamlit</p>
    <p>For clinical decision support only. Not a substitute for professional medical judgment.</p>
</div>
""", unsafe_allow_html=True)