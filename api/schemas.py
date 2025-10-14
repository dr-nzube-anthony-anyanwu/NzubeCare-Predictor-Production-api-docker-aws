from pydantic import BaseModel, Field
from typing import Optional, Literal

class PatientInput(BaseModel):
    # Demographics
    race: Literal["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"] = Field(..., description="Patient race")
    gender: Literal["Female", "Male"] = Field(..., description="Patient gender")
    age_num: int = Field(..., ge=5, le=95, description="Patient age")
    
    # Hospitalization details
    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital")
    num_lab_procedures: int = Field(..., ge=1, le=100, description="Number of lab procedures")
    num_procedures: int = Field(..., ge=0, le=10, description="Number of procedures")
    num_medications: int = Field(..., ge=1, le=30, description="Number of medications")
    number_outpatient: int = Field(..., ge=0, le=20, description="Outpatient visits in past year")
    number_emergency: int = Field(..., ge=0, le=20, description="Emergency visits in past year")
    number_inpatient: int = Field(..., ge=0, le=10, description="Inpatient visits in past year")
    
    # Medical details
    number_diagnoses: int = Field(..., ge=1, le=20, description="Number of diagnoses")
    max_glu_serum: Literal["Norm", ">200", ">300", "None"] = Field(..., description="Max glucose serum level")
    A1Cresult: Literal["Norm", ">7", ">8", "None"] = Field(..., description="A1C test result")
    change: Literal["No", "Ch"] = Field(..., description="Medication change")
    diabetesMed: Literal["No", "Yes"] = Field(..., description="Diabetes medication prescribed")

class PredictionResponse(BaseModel):
    patient_id: Optional[str] = None
    readmission_risk: str
    risk_probability: float
    risk_level: str
    recommendations: list[str]
    feature_importance: dict
    model_version: str = "1.0"

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
