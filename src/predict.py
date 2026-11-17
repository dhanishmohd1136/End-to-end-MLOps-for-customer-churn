
import time
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)



MODEL_NAME = "ChurnModel"

client = MlflowClient()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLflow Prediction, Rollback and Monitoring API",
    version="1.0.0"
)




API_REQUESTS = Counter(
    "churnops_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

API_LATENCY = Histogram(
    "churnops_api_request_duration_seconds",
    "API request processing duration",
    ["endpoint"]
)

PREDICTIONS_TOTAL = Counter(
    "churnops_predictions_total",
    "Total successful predictions",
    ["prediction"]
)

PREDICTION_ERRORS = Counter(
    "churnops_prediction_errors_total",
    "Total failed predictions"
)

ROLLBACK_TOTAL = Counter(
    "churnops_rollbacks_total",
    "Total successful rollback operations"
)

ROLLBACK_ERRORS = Counter(
    "churnops_rollback_errors_total",
    "Total failed rollback operations"
)




class CustomerFeatures(BaseModel):

    Age: int = Field(ge=18, le=100)
    Tenure: int = Field(ge=0)
    SupportCalls: int = Field(ge=0)
    PaymentDelay: int = Field(ge=0)


class PredictionRequest(BaseModel):

    features: CustomerFeatures


class RollbackRequest(BaseModel):

    version: int = Field(gt=0)




@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start_time

    endpoint = request.url.path

    if endpoint != "/metrics":

        API_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code)
        ).inc()

        API_LATENCY.labels(
            endpoint=endpoint
        ).observe(duration)

    return response




@app.post("/predict", tags=["Prediction"])
def predict(request: PredictionRequest):

    try:

        # Load champion model
        model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}@champion"
        )

        # Convert input to DataFrame
        data = pd.DataFrame([
            request.features.model_dump()
        ])

        # Generate prediction
        prediction = model.predict(data)

        result = int(prediction[0])

        # Record prediction metric
        PREDICTIONS_TOTAL.labels(
            prediction=str(result)
        ).inc()

        return {
            "model_name": MODEL_NAME,
            "prediction": result,
            "churn_status": (
                "Churn" if result == 1
                else "No Churn"
            )
        }

    except Exception as e:

        PREDICTION_ERRORS.inc()

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )




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

        # Record successful rollback
        ROLLBACK_TOTAL.inc()

        return {
            "status": "success",
            "message": "Model rollback successful",
            "model_name": MODEL_NAME,
            "champion_version": request.version
        }

    except Exception as e:

        ROLLBACK_ERRORS.inc()

        raise HTTPException(
            status_code=400,
            detail=f"Rollback failed: {str(e)}"
        )




@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():

    return Response(
        content=generate_latest(),
        headers={
            "Content-Type": CONTENT_TYPE_LATEST
        }
    )




@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "endpoints": [
            "POST /predict",
            "POST /rollback",
            "GET /metrics"
        ]
    }
