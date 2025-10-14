from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
import logging

from .schemas import PatientInput, PredictionResponse, HealthResponse
from .utils import (
    load_model, 
    load_expected_columns, 
    preprocess_input, 
    get_risk_level, 
    get_recommendations,
    calculate_feature_importance
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NzubeCare Readmission Predictor API",
    description="API for predicting hospital readmission risk using machine learning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and columns
model = None
expected_columns = None

# Create API router with /api prefix
api_router = APIRouter(prefix="/api", tags=["api"])

@app.on_event("startup")
async def startup_event():
    """Load model and expected columns on startup."""
    global model, expected_columns
    
    try:
        model_path = os.getenv("MODEL_PATH", "ml_models/best_model.pkl")
        columns_path = os.getenv("COLUMNS_PATH", "ml_models/expected_columns.csv")
        
        logger.info(f"Loading model from: {model_path}")
        model = load_model(model_path)
        
        logger.info(f"Loading expected columns from: {columns_path}")
        expected_columns = load_expected_columns(columns_path)
        
        logger.info("Model and columns loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model or columns: {e}")
        # Don't fail startup, but model will be None

# Root endpoint (no prefix)
@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "NzubeCare Readmission Predictor API",
        "version": "1.0.0",
        "docs_url": "/api/docs",
        "health_check": "/health",
        "api_endpoints": "/api/"
    }

# Health check endpoint (no prefix for load balancer)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancer."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        message="API is running" if model is not None else "Model not loaded",
        model_loaded=model is not None
    )

# API endpoints with /api prefix
@api_router.get("/health", response_model=HealthResponse)
async def api_health_check():
    """API health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        message="API is running" if model is not None else "Model not loaded",
        model_loaded=model is not None
    )

@api_router.post("/predict", response_model=PredictionResponse)
async def predict_readmission(patient: PatientInput):
    """Predict readmission risk for a single patient."""
    
    if model is None or expected_columns is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server configuration."
        )
    
    try:
        # Convert Pydantic model to dict
        patient_data = patient.dict()
        
        # Preprocess input data
        processed_data = preprocess_input(patient_data, expected_columns)
        
        # Make prediction
        prediction_proba = model.predict_proba(processed_data)[0]
        readmission_probability = prediction_proba[1]  # Probability of readmission
        
        # Determine risk level
        risk_level = get_risk_level(readmission_probability)
        
        # Generate recommendations
        recommendations = get_recommendations(risk_level, patient_data)
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(model, processed_data, expected_columns)
        
        # Prepare response
        return PredictionResponse(
            readmission_risk=f"{readmission_probability:.3f}",
            risk_probability=round(readmission_probability, 3),
            risk_level=risk_level,
            recommendations=recommendations,
            feature_importance=feature_importance,
            model_version="1.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@api_router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(patients: List[PatientInput]):
    """Predict readmission risk for multiple patients."""
    
    if model is None or expected_columns is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server configuration."
        )
    
    if len(patients) > 100:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large. Maximum 100 patients allowed."
        )
    
    results = []
    
    for i, patient in enumerate(patients):
        try:
            # Convert Pydantic model to dict
            patient_data = patient.dict()
            
            # Preprocess input data
            processed_data = preprocess_input(patient_data, expected_columns)
            
            # Make prediction
            prediction_proba = model.predict_proba(processed_data)[0]
            readmission_probability = prediction_proba[1]
            
            # Determine risk level
            risk_level = get_risk_level(readmission_probability)
            
            # Generate recommendations
            recommendations = get_recommendations(risk_level, patient_data)
            
            # Calculate feature importance
            feature_importance = calculate_feature_importance(model, processed_data, expected_columns)
            
            # Prepare response
            result = PredictionResponse(
                patient_id=f"patient_{i+1}",
                readmission_risk=f"{readmission_probability:.3f}",
                risk_probability=round(readmission_probability, 3),
                risk_level=risk_level,
                recommendations=recommendations,
                feature_importance=feature_importance,
                model_version="1.0"
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Prediction error for patient {i+1}: {e}")
            # Continue with other patients, but log the error
            continue
    
    return results

@api_router.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "model_version": "1.0",
        "features_count": len(expected_columns) if expected_columns else 0,
        "expected_features": expected_columns[:10] if expected_columns else [],  # Show first 10
        "description": "LightGBM model for hospital readmission prediction"
    }

# Include the API router
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
