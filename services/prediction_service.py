import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

FEATURES_PATH = "house_features.joblib"
ONNX_MODEL_PATH = "house_model.onnx"
CONFIDENCE_USD = 39000

features = joblib.load(FEATURES_PATH)
_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name


class MissingColumnsError(Exception):
    def __init__(self, missing_columns):
        self.missing_columns = missing_columns
        super().__init__(f"missing columns: {missing_columns}")


class EmptyFileError(Exception):
    pass


def get_features():
    return features


def _run_onnx(input_rows: np.ndarray) -> np.ndarray:
    onnx_input = input_rows.astype(np.float32)
    return _session.run(None, {_input_name: onnx_input})[0].reshape(-1)


def predict_one(house_data: dict) -> dict:
    input_data = pd.DataFrame([house_data], columns=features)
    predicted = float(_run_onnx(input_data.to_numpy())[0])
    price_usd = predicted * 100000

    return {
        "predicted_price": f"${price_usd:,.0f}",
        "predicted_price_short": f"{predicted:.2f} hundred thousands",
        "confidence_range": f"${price_usd - CONFIDENCE_USD:,.0f} to ${price_usd + CONFIDENCE_USD:,.0f}",
    }


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = list(features)

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise MissingColumnsError(missing_columns)

    if len(df) == 0:
        raise EmptyFileError()

    predictions = _run_onnx(df[required_columns].to_numpy())
    result_df = df.copy()
    result_df["predicted_price_usd"] = predictions * 100000
    result_df["predicted_price"] = result_df["predicted_price_usd"].apply(lambda x: f"${x:,.0f}")

    return result_df
