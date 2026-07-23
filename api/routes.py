import io
import time

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from schemas.house import HouseFeatures, PredictionResponse
from services import job_service, prediction_service

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "California house prediction api",
        "status": "running",
        "endpoint": "send POST request to /predict",
    }


@router.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": prediction_service.get_features(),
        "avg_error": "$39000",
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    start = time.perf_counter()
    result = prediction_service.predict_one(house.model_dump())
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(f"/predict inference latency={latency_ms:.2f}ms")
    return result


@router.post("/predict-file", status_code=202)
async def predict_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="please upload a CSV File only")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV file: {str(e)}")

    df.columns = [col.strip() for col in df.columns]

    job_id = job_service.create_job()
    background_tasks.add_task(job_service.run_prediction_job, job_id, df)

    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if job["status"] == "complete":
        return StreamingResponse(
            io.StringIO(job["result_csv"]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    return {"job_id": job["job_id"], "status": job["status"], "error": job["error"]}
