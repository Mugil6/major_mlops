import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, split_data, evaluate_model
from sklearn.metrics import r2_score, mean_squared_error


def quantize_coefficients_uint8(arr):

    scales = np.zeros_like(arr, dtype=np.float32)
    q = np.zeros_like(arr, dtype=np.uint8)
    for i, val in enumerate(arr):
        max_abs = max(abs(val), 1e-8)
        scale = max_abs / 127.0
        q_val = int(round(val / scale)) + 128
        q_val = max(0, min(255, q_val))
        q[i] = q_val
        scales[i] = scale
    return q, scales


def dequantize_coefficients_uint8(q, scales):

    return (q.astype(np.int16) - 128) * scales


def memory_size(arr):
    return arr.nbytes / 1024


def main():

    model: LinearRegression = joblib.load("model.joblib")
    coef = model.coef_.astype(np.float32)
    intercept = np.array([model.intercept_], dtype=np.float32)


    joblib.dump({"coef": coef, "intercept": intercept}, "unquant_params.joblib")


    q_coef, coef_scales = quantize_coefficients_uint8(coef)
    q_intercept, intercept_scales = quantize_coefficients_uint8(intercept)

    joblib.dump({
        "q_coef": q_coef,
        "coef_scales": coef_scales,
        "q_intercept": q_intercept,
        "intercept_scales": intercept_scales
    }, "quant_params.joblib")


    orig_mem = memory_size(coef) + memory_size(intercept)
    quant_mem = memory_size(q_coef) + memory_size(q_intercept)
    print("\n=== Memory Usage ===")
    print(f"Original: {orig_mem:.2f} KB")
    print(f"Quantized: {quant_mem:.2f} KB")


    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    r2_orig, mse_orig = evaluate_model(model, X_test, y_test)


    deq_coef = dequantize_coefficients_uint8(q_coef, coef_scales)
    deq_intercept = dequantize_coefficients_uint8(q_intercept, intercept_scales)[0]
    preds = X_test @ deq_coef + deq_intercept

    r2_q = r2_score(y_test, preds)
    mse_q = mean_squared_error(y_test, preds)

    print("\n=== Evaluation Metrics ===")
    print(f"Original -> R2: {r2_orig:.4f}, MSE: {mse_orig:.4f}")
    print(f"Quantized (uint8) -> R2: {r2_q:.4f}, MSE: {mse_q:.4f}")

    print("\nSample quantized coefficients:", q_coef[:5])
    print("Sample predictions:", preds[:5])


if __name__ == "__main__":
    main()
