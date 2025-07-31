import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import load_data, split_data, evaluate_model
from sklearn.metrics import r2_score, mean_squared_error

def quantize_to_float16(arr):

    return arr.astype(np.float16)

def dequantize_from_float16(arr):

    return arr.astype(np.float32)

def memory_size(arr):

    return arr.nbytes / 1024

def main():

    model: LinearRegression = joblib.load("model.joblib")


    coef = model.coef_
    intercept = model.intercept_


    joblib.dump({"coef": coef, "intercept": intercept}, "unquant_params.joblib")
    print("Unquantized parameters saved.")


    q_coef = quantize_to_float16(coef)
    q_intercept = quantize_to_float16(np.array([intercept]))

    quant_data = {"q_coef": q_coef, "q_intercept": q_intercept}
    joblib.dump(quant_data, "quant_params.joblib")
    print("Float16 quantized parameters saved.")


    print("\n=== Parameters Comparison (first 5 coefficients) ===")
    print("Original (float64):", coef[:5])
    print("Quantized (float16):", q_coef[:5])


    original_mem = memory_size(coef) + memory_size(np.array([intercept]))
    quantized_mem = memory_size(q_coef) + memory_size(q_intercept)
    print("\n=== Memory Usage ===")
    print(f"Original params memory: {original_mem:.2f} KB")
    print(f"Quantized params memory: {quantized_mem:.2f} KB")


    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    r2_orig, mse_orig = evaluate_model(model, X_test, y_test)

    deq_coef = dequantize_from_float16(q_coef)
    deq_intercept = dequantize_from_float16(q_intercept)[0]

    preds = X_test @ deq_coef + deq_intercept


    r2_quant = r2_score(y_test, preds)
    mse_quant = mean_squared_error(y_test, preds)

    print("\n=== Evaluation Metrics ===")
    print(f"Original Model -> R2: {r2_orig:.4f}, MSE: {mse_orig:.4f}")
    print(f"Quantized Model (float16) -> R2: {r2_quant:.4f}, MSE: {mse_quant:.4f}")

    print("\nSample predictions with quantized model:", preds[:5])

if __name__ == "__main__":
    main()
