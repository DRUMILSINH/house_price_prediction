import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from services import prediction_service

_jobs: dict[str, dict] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_csv": None,
        "error": None,
    }
    return job_id


def get_job(job_id: str) -> dict | None:
    return _jobs.get(job_id)


def run_prediction_job(job_id: str, df: pd.DataFrame) -> None:
    start = time.perf_counter()
    try:
        result_df = prediction_service.predict_dataframe(df)
        _jobs[job_id]["result_csv"] = result_df.to_csv(index=False)
        _jobs[job_id]["status"] = "complete"
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"job {job_id} completed: {len(df)} rows, inference latency={latency_ms:.2f}ms")

    except prediction_service.MissingColumnsError as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = f"These columns are missing from your file{e.missing_columns}"
        logger.warning(f"job {job_id} rejected: missing columns {e.missing_columns}")

    except prediction_service.EmptyFileError:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = "The upload file has no data rows"
        logger.warning(f"job {job_id} rejected: empty file")

    except Exception:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = "Prediction failed due to an internal error"
        logger.opt(exception=True).error(f"job {job_id} failed with an unexpected error")
