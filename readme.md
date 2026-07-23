# California House Price Prediction API

A simple FastAPI project that predicts California house prices using a trained `RandomForestRegressor` model from scikit-learn.

## Features

- Single prediction endpoint using JSON input
- Batch prediction endpoint using CSV upload
- Model training script using the California Housing dataset
- Health endpoint for quick API checks

## Project Structure

```text
.
|-- main.py
|-- train.py
|-- explore.py
|-- test_houses.csv
|-- readme.md
|-- .gitignore
```

## Tech Stack

- Python
- FastAPI
- pandas
- scikit-learn
- joblib
- Uvicorn

## Setup

1. Create and activate a virtual environment.
2. Install the required packages.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn pandas scikit-learn joblib python-multipart
```

## Train The Model

Run the training script to generate the local model files:

```powershell
python train.py
```

This creates:

- `house_model.joblib`
- `house_features.joblib`

These files are ignored by Git because `house_model.joblib` is too large for a standard GitHub push.

## Run The API

```powershell
uvicorn main:app --reload
```

API docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## API Endpoints

### `GET /`

Returns a basic project status message.

### `GET /health`

Returns API health and loaded model metadata.

### `POST /predict`

Predict a single house price using JSON input.

Example request body:

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

### `POST /predict-file`

Upload a CSV file with these columns:

- `MedInc`
- `HouseAge`
- `AveRooms`
- `AveBedrms`
- `Population`
- `AveOccup`
- `Latitude`
- `Longitude`

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/predict-file" `
  -H "accept: text/csv" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@test_houses.csv"
```

The API returns a CSV file with prediction columns appended.

## Notes

- `test_houses.csv` is included as a sample upload file.
- `predictions.csv` is ignored because it is generated output.
- If you want to version model artifacts, use Git LFS instead of standard Git.
