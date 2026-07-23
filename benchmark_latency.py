import time

import joblib
import numpy as np
import onnxruntime as ort

FEATURES_PATH = "house_features.joblib"
MODEL_PATH = "house_model.joblib"
ONNX_PATH = "house_model.onnx"
N = 1000


def main():
    sklearn_model = joblib.load(MODEL_PATH)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(42)
    rows = np.column_stack(
        [
            rng.uniform(0.5, 15, N),  # MedInc
            rng.uniform(1, 52, N),  # HouseAge
            rng.uniform(2, 10, N),  # AveRooms
            rng.uniform(0.8, 2, N),  # AveBedrms
            rng.uniform(3, 5000, N),  # Population
            rng.uniform(1, 6, N),  # AveOccup
            rng.uniform(32, 42, N),  # Latitude
            rng.uniform(-125, -114, N),  # Longitude
        ]
    ).astype(np.float32)

    # warmup, so first-call overhead doesn't skew the timed loop
    sklearn_model.predict(rows[:5].astype(np.float64))
    session.run(None, {input_name: rows[:5]})

    start = time.perf_counter()
    for row in rows:
        sklearn_model.predict(row.reshape(1, -1).astype(np.float64))
    sklearn_total = time.perf_counter() - start

    start = time.perf_counter()
    for row in rows:
        session.run(None, {input_name: row.reshape(1, -1)})
    onnx_total = time.perf_counter() - start

    print(f"predictions: {N}")
    print(f"sklearn: {sklearn_total:.3f}s total -> {sklearn_total / N * 1000:.4f} ms/prediction")
    print(f"onnx:    {onnx_total:.3f}s total -> {onnx_total / N * 1000:.4f} ms/prediction")
    print(f"speedup: {sklearn_total / onnx_total:.2f}x")


if __name__ == "__main__":
    main()
