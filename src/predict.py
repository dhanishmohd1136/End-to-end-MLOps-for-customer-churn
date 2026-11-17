
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "ChurnModel"

client = MlflowClient()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLflow Model Prediction and Rollback API",
    version="1.0.0"
)


# ==========================================
# INPUT VALIDATION
# ==========================================

class CustomerFeatures(BaseModel):

    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0)
    SupportCalls: int = Field(ge=0)
    PaymentDelay: int = Field(ge=0)


class PredictionRequest(BaseModel):

    features: CustomerFeatures


class RollbackRequest(BaseModel):

    version: int = Field(gt=0)


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict", tags=["Prediction"])
def predict(request: PredictionRequest):

    try:

        # Load champion model from MLflow
        model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        # Convert input into DataFrame
        data = pd.DataFrame([
            request.features.model_dump()
        ])

        # Generate prediction
        prediction = model.predict(data)

        result = int(prediction[0])

        return {
            "model_name": MODEL_NAME,
            "prediction": result,
            "churn_status": (
                "Churn" if result == 1
                else "No Churn"
            )
        }

    except MlflowException as e:

        raise HTTPException(
            status_code=503,
            detail=f"MLflow model error: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ==========================================
# ROLLBACK ENDPOINT
# ==========================================

@app.post("/rollback", tags=["Model Management"])
def rollback(request: RollbackRequest):

    try:

        # Validate model version
        client.get_model_version(
            name=MODEL_NAME,
            version=str(request.version)
        )

        # Update champion alias
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=str(request.version)
        )

        return {
            "status": "success",
            "message": "Model rollback successful",
            "model_name": MODEL_NAME,
            "champion_version": request.version
        }

    except MlflowException as e:

        raise HTTPException(
            status_code=404,
            detail=f"MLflow error: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "endpoints": [
            "POST /predict",
            "POST /rollback"
        ],
        "documentation": "/docs"
    }
