import joblib
import numpy as np
import onnxruntime as ort
import pytest

MODEL_PATH = "house_model.joblib"
FEATURES_PATH = "house_features.joblib"
ONNX_PATH = "house_model.onnx"

# A handful of realistic California-housing rows, independent of the train/test split.
SAMPLE_INPUTS = np.array(
    [
        [8.3252, 41, 6.984, 1.023, 322, 2.555, 37.88, -122.23],
        [2.5, 30, 4.5, 1.2, 1200, 3.5, 34.05, -118.25],
        [5.0, 15, 7.0, 1.5, 800, 2.8, 36.5, -120.0],
        [3.8, 20, 5.5, 1.1, 950, 3.0, 35.0, -119.5],
        [9.5, 10, 8.0, 2.0, 400, 2.2, 37.78, -122.41],
    ],
    dtype=np.float32,
)


@pytest.fixture(scope="module")
def sklearn_model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def onnx_session():
    return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])


def test_onnx_predictions_match_sklearn_within_tolerance(sklearn_model, onnx_session):
    sklearn_preds = sklearn_model.predict(SAMPLE_INPUTS.astype(np.float64))

    input_name = onnx_session.get_inputs()[0].name
    onnx_preds = onnx_session.run(None, {input_name: SAMPLE_INPUTS})[0].reshape(-1)

    np.testing.assert_allclose(onnx_preds, sklearn_preds, atol=1e-3)
