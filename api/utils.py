import pandas as pd
import joblib
import numpy as np
from typing import Dict, List, Any

def load_model(model_path: str):
    """Load the trained model from file."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

def load_expected_columns(columns_path: str) -> List[str]:
    """Load expected column names from CSV file."""
    try:
        expected_cols_df = pd.read_csv(columns_path)
        if len(expected_cols_df.columns) == 1:
            return expected_cols_df.iloc[:, 0].tolist()
        else:
            return expected_cols_df.columns.tolist()
    except Exception as e:
        # Return default columns if file not found
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

def preprocess_input(data: Dict, expected_columns: List[str]) -> pd.DataFrame:
    """Preprocess input data to match model training format."""
    
    # Create base DataFrame with numerical features
    processed_data = {
        'time_in_hospital': data['time_in_hospital'],
        'num_lab_procedures': data['num_lab_procedures'],
        'num_procedures': data['num_procedures'],
        'num_medications': data['num_medications'],
        'number_outpatient': data['number_outpatient'],
        'number_emergency': data['number_emergency'],
        'number_inpatient': data['number_inpatient'],
        'number_diagnoses': data['number_diagnoses'],
        'age_num': data['age_num']
    }
    
    # One-hot encode race
    races = ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other']
    for race in races:
        processed_data[f'race_{race}'] = 1 if data['race'] == race else 0
    
    # One-hot encode gender
    processed_data['gender_Female'] = 1 if data['gender'] == 'Female' else 0
    processed_data['gender_Male'] = 1 if data['gender'] == 'Male' else 0
    
    # One-hot encode max_glu_serum
    glu_levels = ['>200', '>300', 'Norm', 'None']
    for level in glu_levels:
        processed_data[f'max_glu_serum_{level}'] = 1 if data['max_glu_serum'] == level else 0
    
    # One-hot encode A1Cresult
    a1c_levels = ['>7', '>8', 'Norm', 'None']
    for level in a1c_levels:
        processed_data[f'A1Cresult_{level}'] = 1 if data['A1Cresult'] == level else 0
    
    # One-hot encode change
    processed_data['change_Ch'] = 1 if data['change'] == 'Ch' else 0
    processed_data['change_No'] = 1 if data['change'] == 'No' else 0
    
    # One-hot encode diabetesMed
    processed_data['diabetesMed_No'] = 1 if data['diabetesMed'] == 'No' else 0
    processed_data['diabetesMed_Yes'] = 1 if data['diabetesMed'] == 'Yes' else 0
    
    # Convert to DataFrame
    input_df = pd.DataFrame([processed_data])
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    return input_df[expected_columns]

def get_risk_level(probability: float) -> str:
    """Determine risk level based on probability."""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"

def get_recommendations(risk_level: str, patient_data: Dict) -> List[str]:
    """Generate recommendations based on risk level and patient data."""
    recommendations = []
    
    if risk_level == "High":
        recommendations.extend([
            "Schedule follow-up appointment within 7 days",
            "Consider discharge planning with care coordinator",
            "Provide detailed medication reconciliation",
            "Arrange home health services if appropriate"
        ])
    elif risk_level == "Medium":
        recommendations.extend([
            "Schedule follow-up appointment within 14 days",
            "Provide clear discharge instructions",
            "Ensure patient understands medication regimen"
        ])
    else:
        recommendations.extend([
            "Standard discharge planning",
            "Schedule routine follow-up as appropriate"
        ])
    
    # Add specific recommendations based on patient characteristics
    if patient_data['diabetesMed'] == 'Yes':
        recommendations.append("Monitor blood glucose levels closely")
    
    if patient_data['number_inpatient'] > 1:
        recommendations.append("Review care coordination to prevent frequent readmissions")
    
    if patient_data['age_num'] > 65:
        recommendations.append("Consider geriatric care consultation")
    
    return recommendations

def calculate_feature_importance(model, input_data: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    """Calculate and return feature importance for the prediction."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances.tolist()))
            # Sort by importance and return top 10
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features[:10])
        else:
            return {"feature_importance": "not_available"}
    except Exception:
        return {"feature_importance": "calculation_failed"}
