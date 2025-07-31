# MLOps Major Assignment – Linear Regression Pipeline

## Overview
This project implements a **complete MLOps pipeline** using the **California Housing dataset** and **Linear Regression**
It includes model training, testing, quantization (float16), Dockerization, and CI/CD using GitHub Actions.

The entire workflow runs in a single branch (`main`) as per the guidelines.

---

## Features
1. **Training**
   - Loads California Housing dataset
   - Trains a `LinearRegression` model
   - Saves the model (`model.joblib`)

2. **Testing**
   - Pytest checks for:
     - Dataset loading
     - Model type and coefficients
     - Minimum R² threshold (>0.5)

3. **Quantization (float16)**
   - Parameters (coefficients & intercept) are saved in:
     - `unquant_params.joblib`
     - `quant_params.joblib`
   - Quantization reduces memory usage (float64 → float16) with negligible accuracy loss.

4. **Dockerization** 
   - Dockerfile builds a container that runs `predict.py` inside.

5. **CI/CD Workflow** 
   - **3 jobs:** test → train & quantize → build & run Docker 
   - Automatically triggered on every push to `main`.

## Results

### Model Performance

| Metric                | Original Model (float64) | Quantized Model (float16) |
|-----------------------|---------------------------|----------------------------|
| R² Score             | 0.576                     | 0.576                     |
| MSE                  | 0.556                     | 0.556                     |
| Params Memory (KB)   | 0.07                      | 0.02                      |



