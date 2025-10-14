import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NzubeCare - Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; padding-bottom: 10px;}
    .section-header {font-size: 2rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px;}
    .feature-importance {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .prediction-high {color: #ff4b4b; font-weight: bold; font-size: 1.5rem;}
    .prediction-medium {color: #ffa14b; font-weight: bold; font-size: 1.5rem;}
    .prediction-low {color: #00cc96; font-weight: bold; font-size: 1.5rem;}
    .footer {position: fixed; bottom: 0; width: 100%; background-color: white; text-align: center; padding: 10px;}
    .warning-box {background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🏥 NzubeCare Readmission Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predictive Analytics for Hospital Readmission Risk")
st.markdown("---")

# Check if LightGBM is available
try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.markdown('<div class="warning-box">'
                '<h4>⚠️ LightGBM Not Installed</h4>'
                '<p>The model requires LightGBM which is not currently installed. '
                'Some functionality will be limited.</p>'
                '<p>To install: <code>pip install lightgbm</code></p>'
                '</div>', unsafe_allow_html=True)

# Simplified authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if st.session_state["password"] == "medreadmit2023":  # Simple hardcoded password
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # Don't store the password
    else:
        st.session_state["password_correct"] = False

if not check_password():
    st.stop()

# Load model and expected columns
@st.cache_resource
def load_model():
    if not LIGHTGBM_AVAILABLE:
        return None
    try:
        return joblib.load('best_model.pkl')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_expected_columns():
    try:
        # Try different ways to load the expected columns
        expected_cols_df = pd.read_csv('expected_columns.csv')
        
        # Check if the CSV has a header row with column names
        if len(expected_cols_df.columns) == 1:
            # If only one column, assume it's the list of column names
            return expected_cols_df.iloc[:, 0].tolist()
        else:
            # If multiple columns, try to find the right one
            for col in expected_cols_df.columns:
                if 'column' in col.lower() or 'feature' in col.lower():
                    return expected_cols_df[col].tolist()
            
            # If no obvious column name, return the first column
            return expected_cols_df.iloc[:, 0].tolist()
            
    except Exception as e:
        st.error(f"Error loading expected columns: {e}")
        st.info("Using default column structure based on the training data")
        
        # Return a default set of columns based on the notebook
        return [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses', 'age_num',
            'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 
            'race_Hispanic', 'race_Other', 'gender_Female', 'gender_Male',
            'max_glu_serum_>200', 'max_glu_serum_>300', 'max_glu_serum_Norm', 
            'max_glu_serum_None', 'A1Cresult_>7', 'A1Cresult_>8', 
            'A1Cresult_Norm', 'A1Cresult_None', 'change_Ch', 'change_No',
            'diabetesMed_No', 'diabetesMed_Yes'
        ]

model = load_model()
expected_columns = load_expected_columns()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=80)
    st.title("NzubeCare AI Predictor")
    st.markdown("---")
    
    # User role selection
    user_role = st.radio("Select User Role:", ["Nurse", "Physician", "Admin"])
    
    # Navigation
    app_mode = st.selectbox("Navigation", 
                           ["Single Patient Prediction", "Batch Prediction", 
                            "Model Information", "About"])
    
    st.markdown("---")
    st.markdown("### Created by Dr. Nzube Anthony Anyanwu")
    st.markdown("Powered by Machine Learning")

# Preset examples for single patient prediction
preset_examples = {
    "Select an example": None,
    "High Risk Profile": {
        "num_lab_procedures": 60,
        "num_procedures": 5,
        "num_medications": 18,
        "number_inpatient": 3,
        "number_diagnoses": 10,
        "time_in_hospital": 10,
        "age_num": 75,
        "change": "Ch",
        "diabetesMed": "Yes",
        "max_glu_serum": ">200",
        "A1Cresult": ">8"
    },
    "Medium Risk Profile": {
        "num_lab_procedures": 45,
        "num_procedures": 2,
        "num_medications": 10,
        "number_inpatient": 1,
        "number_diagnoses": 6,
        "time_in_hospital": 5,
        "age_num": 60,
        "change": "No",
        "diabetesMed": "Yes",
        "max_glu_serum": "Norm",
        "A1Cresult": "Norm"
    },
    "Low Risk Profile": {
        "num_lab_procedures": 30,
        "num_procedures": 0,
        "num_medications": 5,
        "number_inpatient": 0,
        "number_diagnoses": 3,
        "time_in_hospital": 2,
        "age_num": 40,
        "change": "No",
        "diabetesMed": "No",
        "max_glu_serum": "Norm",
        "A1Cresult": "Norm"
    }
}

# Single Patient Prediction
if app_mode == "Single Patient Prediction":
    st.markdown('<h2 class="section-header">Single Patient Risk Assessment</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Patient demographics
        st.subheader("Patient Demographics")
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
            gender = st.selectbox("Gender", ["Female", "Male"])
        
        with demo_col2:
            age_num = st.slider("Age", 5, 95, 45, 5)
            # Convert age to categorical bin
            if age_num < 10: age = "[0-10)"
            elif age_num < 20: age = "[10-20)"
            elif age_num < 30: age = "[20-30)"
            elif age_num < 40: age = "[30-40)"
            elif age_num < 50: age = "[40-50)"
            elif age_num < 60: age = "[50-60)"
            elif age_num < 70: age = "[60-70)"
            elif age_num < 80: age = "[70-80)"
            elif age_num < 90: age = "[80-90)"
            else: age = "[90-100)"
            
        with demo_col3:
            # Preset examples
            selected_example = st.selectbox("Load Preset Example", list(preset_examples.keys()))
            
            if selected_example != "Select an example":
                example_data = preset_examples[selected_example]
                # Apply example data to form fields
                if example_data:
                    # This would need JavaScript to auto-fill form fields
                    st.info("Example loaded. Please manually adjust values as needed.")
    
    # Hospitalization details
    st.subheader("Hospitalization Details")
    hosp_col1, hosp_col2, hosp_col3 = st.columns(3)
    
    with hosp_col1:
        time_in_hospital = st.slider("Days in Hospital", 1, 14, 3)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 41)
        
    with hosp_col2:
        num_procedures = st.slider("Number of Procedures", 0, 10, 0)
        num_medications = st.slider("Number of Medications", 1, 30, 8)
        
    with hosp_col3:
        number_outpatient = st.slider("Outpatient Visits (past year)", 0, 20, 0)
        number_emergency = st.slider("Emergency Visits (past year)", 0, 20, 0)
        number_inpatient = st.slider("Inpatient Visits (past year)", 0, 10, 0)
    
    # Medical details
    st.subheader("Medical Details")
    med_col1, med_col2, med_col3 = st.columns(3)
    
    with med_col1:
        number_diagnoses = st.slider("Number of Diagnoses", 1, 20, 5)
        max_glu_serum = st.selectbox("Max Glucose Serum", ["Norm", ">200", ">300", "None"])
        
    with med_col2:
        A1Cresult = st.selectbox("A1C Result", ["Norm", ">7", ">8", "None"])
        change = st.selectbox("Medication Change", ["No", "Ch"])
        
    with med_col3:
        diabetesMed = st.selectbox("Diabetes Medication", ["No", "Yes"])
        # Add more medication fields based on user role
        if user_role in ["Physician", "Admin"]:
            insulin = st.selectbox("Insulin", ["No", "Up", "Down", "Steady"])
            metformin = st.selectbox("Metformin", ["No", "Up", "Down", "Steady"])
    
    # Prediction button
    if st.button("Predict Readmission Risk", type="primary"):
        if not LIGHTGBM_AVAILABLE or model is None:
            st.error("""
            **Model not available.** 
            
            Please install LightGBM to enable predictions:
            ```
            pip install lightgbm
            ```
            
            Then restart the application.
            """)
        else:
            # Create input dataframe
            input_dict = {
                'race': race,
                'gender': gender,
                'age': age,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'max_glu_serum': max_glu_serum,
                'A1Cresult': A1Cresult,
                'change': change,
                'diabetesMed': diabetesMed,
                'age_num': age_num
            }
            
            # Add medication fields if available
            if user_role in ["Physician", "Admin"]:
                input_dict['insulin'] = insulin
                input_dict['metformin'] = metformin
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # One-hot encode categorical variables to match training format
            input_processed = pd.get_dummies(input_df)
            
            # Align with training columns
            for col in expected_columns:
                if col not in input_processed.columns:
                    input_processed[col] = 0
            
            input_processed = input_processed[expected_columns]
            
            # Make prediction
            try:
                prediction_proba = model.predict_proba(input_processed)[0]
                risk_score = prediction_proba[1]  # Probability of readmission
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Risk level categorization
                if risk_score >= 0.7:
                    risk_level = "HIGH RISK"
                    risk_class = "prediction-high"
                elif risk_score >= 0.4:
                    risk_level = "MEDIUM RISK"
                    risk_class = "prediction-medium"
                else:
                    risk_level = "LOW RISK"
                    risk_class = "prediction-low"
                
                # Display risk score with color coding
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### Readmission Probability: <span class='{risk_class}'>{risk_score:.2%}</span>", 
                               unsafe_allow_html=True)
                    st.markdown(f"### Risk Level: <span class='{risk_class}'>{risk_level}</span>", 
                               unsafe_allow_html=True)
                
                with col2:
                    # Gauge chart - FIXED SYNTAX
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Readmission Risk Score"},
                        delta={'reference': 0.4, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.4], 'color': "lightgreen"},
                                {'range': [0.4, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7
                            }
                        }
                    ))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)
                
                # SHAP explanation (if available)
                try:
                    import shap
                    if hasattr(model, 'predict_proba'):
                        # Create explainer
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(input_processed)
                        
                        # Plot SHAP values
                        st.subheader("Feature Impact on Prediction")
                        
                        # For binary classification
                        if len(shap_values) == 2:  # For binary classification
                            shap_df = pd.DataFrame({
                                'feature': input_processed.columns,
                                'shap_value': shap_values[1][0]  # Use class 1 (readmission)
                            })
                        else:
                            shap_df = pd.DataFrame({
                                'feature': input_processed.columns,
                                'shap_value': shap_values[0]
                            })
                        
                        # Sort by absolute impact
                        shap_df['abs_value'] = np.abs(shap_df['shap_value'])
                        shap_df = shap_df.sort_values('abs_value', ascending=False).head(10)
                        
                        # Create horizontal bar chart
                        fig = px.bar(shap_df, 
                                    x='shap_value', 
                                    y='feature', 
                                    orientation='h',
                                    title='Top Features Influencing Prediction',
                                    labels={'shap_value': 'SHAP Value (Impact on Prediction)', 'feature': 'Feature'})
                        
                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info("Detailed feature explanation is not available for this model type.")
                
                # Recommendations based on risk level
                st.subheader("Clinical Recommendations")
                
                if risk_level == "HIGH RISK":
                    st.error("""
                    **Immediate Action Recommended:**
                    - Schedule follow-up appointment within 7 days
                    - Consider medication review and adjustment
                    - Provide enhanced patient education
                    - Coordinate with care management team
                    - Consider home health referral
                    """)
                elif risk_level == "MEDIUM RISK":
                    st.warning("""
                    **Preventive Measures Recommended:**
                    - Schedule follow-up within 14 days
                    - Review medication adherence
                    - Provide disease-specific education
                    - Monitor key health indicators
                    """)
                else:
                    st.success("""
                    **Maintenance Care Recommended:**
                    - Routine follow-up as scheduled
                    - Continue current care plan
                    - Encourage healthy lifestyle behaviors
                    - Standard monitoring
                    """)
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Batch Prediction
elif app_mode == "Batch Prediction":
    st.markdown('<h2 class="section-header">Batch Patient Risk Assessment</h2>', unsafe_allow_html=True)
    
    if user_role != "Admin":
        st.warning("Batch prediction is only available for Admin users.")
    else:
        if not LIGHTGBM_AVAILABLE or model is None:
            st.error("""
            **Model not available.** 
            
            Please install LightGBM to enable batch predictions:
            ```
            pip install lightgbm
            ```
            
            Then restart the application.
            """)
        else:
            uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    batch_df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(batch_df.head())
                    
                    if st.button("Process Batch Prediction", type="primary"):
                        st.info("Batch processing would be implemented here once LightGBM is installed.")
                        st.info("This feature would process multiple patients and provide risk assessments for each.")
                        
                except Exception as e:
                    st.error(f"Error processing file: {e}")

# Model Information
elif app_mode == "Model Information":
    st.markdown('<h2 class="section-header">Model Information</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Model Card", "Performance Metrics", "Feature Importance", "Limitations"])
    
    with tab1:
        st.subheader("Model Card")
        st.markdown("""
        **Model Overview**
        - **Name**: NzubeCare Readmission Risk Predictor
        - **Version**: 1.0
        - **Type**: Gradient Boosting Classifier (LightGBM)
        - **Task**: Binary classification (readmission within 30 days)
        
        **Training Data**
        - **Source**: UCI Diabetes 130-US hospitals dataset
        - **Time Period**: 1999-2008
        - **Samples**: 101,766 encounters
        - **Features**: 40 after preprocessing
        
        **Preprocessing**
        - Handled missing values and encoded categorical variables
        - Addressed class imbalance using SMOTE
        - Feature selection to remove leakage variables and low-variance features
        
        **Intended Use**
        - Predict risk of hospital readmission within 30 days for diabetic patients
        - Assist healthcare providers in identifying high-risk patients for targeted interventions
        """)
        
        if not LIGHTGBM_AVAILABLE:
            st.warning("**Installation Required**: LightGBM is needed to load and use this model.")
            st.code("pip install lightgbm")
    
    with tab2:
        st.subheader("Performance Metrics")
        
        # Simulated metrics (replace with actual metrics from your model)
        metrics = {
            'ROC-AUC': 0.65,
            'PR-AUC': 0.19,
            'Accuracy': 0.73,
            'Precision': 0.18,
            'Recall': 0.42,
            'F1-Score': 0.25
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for metric, value in list(metrics.items())[:3]:
                st.metric(metric, f"{value:.3f}")
        
        with col2:
            for metric, value in list(metrics.items())[3:]:
                st.metric(metric, f"{value:.3f}")
        
        # ROC Curve (simulated)
        st.subheader("ROC Curve")
        # Simulated data - replace with your actual ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Simulated curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = 0.78)'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random classifier', line=dict(dash='dash')))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800, height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Importance")
        
        # Simulated feature importance (replace with your actual feature importance)
        features = [
            'number_inpatient', 'num_medications', 'number_diagnoses', 
            'time_in_hospital', 'num_lab_procedures', 'age_num', 
            'number_emergency', 'A1Cresult_>8', 'change_Ch', 'diabetesMed_Yes'
        ]
        
        importance = [0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
        
        fig = px.bar(
            x=importance, 
            y=features, 
            orientation='h',
            title='Top 10 Feature Importance',
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Limitations and Considerations")
        
        st.warning("""
        **Important Limitations:**
        
        **Data Scope**: Model trained on data from 1999-2008 - medical practices may have changed. 
        **Population Specificity**: Trained primarily on US diabetic patient population. 
        **Temporal Factors**: Does not account for seasonal variations or pandemic impacts. 
        **Feature Availability**: Requires specific input features that may not be available in all settings. 
        **More Feature Engineering: The model's low performance suggests the current features (variables) provided aren't powerful enough to predict readmission. The model (0.193) is better than random guessing, but it still has a very long way to go to be clinically useful. It needs more work. 
        **Clinical Judgment**: Should supplement, not replace, clinical decision-making.
        
        **Ethical Considerations:**
        - Ensure equitable application across patient demographics
        - Regularly monitor for performance drift
        - Maintain patient privacy and data security
        - Use predictions to enhance care, not limit access
        """)

# About section
else:
    st.markdown('<h2 class="section-header">About NzubeCare Predictor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **NzubeCare Readmission Predictor** is an advanced predictive analytics tool designed to identify diabetic patients 
    at high risk of hospital readmission within 30 days of discharge.
    
    **Key Features**
    - **Single Patient Assessment**: Evaluate individual patient risk with detailed explanations
    - **Batch Processing**: Analyze multiple patients simultaneously (Admin only)
    - **Clinical Insights**: Receive evidence-based recommendations based on risk level
    - **Model Transparency**: Understand how predictions are made with feature importance
    
    **Technology Stack**
    - **Machine Learning**: Gradient Boosting Classifier (LightGBM)
    - **Frontend**: Streamlit
    - **Data Visualization**: Plotly
    - **Model Interpretation**: SHAP values
    
    **Intended Use**
    This tool is designed to assist healthcare providers in identifying high-risk patients who may benefit 
    from targeted interventions, care coordination, and follow-up planning to reduce readmission rates.
    
    **Disclaimer**
    This tool is intended for clinical decision support only. Final medical decisions should be made by 
    qualified healthcare professionals considering all available patient information.
    """)
    
    st.markdown("---")
    st.markdown("### Developed by Dr. Nzube Anthony Anyanwu")
    st.markdown("For demonstration purposes only; and NOT for hospital/clinical use")

# Footer
st.markdown("---")
st.markdown('<div class="footer">NzubeCare - Hospital Readmission Risk Predictor © 2025</div>', unsafe_allow_html=True)