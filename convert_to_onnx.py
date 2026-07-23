import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_PATH = "house_model.joblib"
FEATURES_PATH = "house_features.joblib"
ONNX_PATH = "house_model.onnx"


def main():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    initial_type = [("input", FloatTensorType([None, len(features)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=17)

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Saved ONNX model to {ONNX_PATH}")


if __name__ == "__main__":
    main()
