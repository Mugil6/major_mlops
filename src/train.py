import joblib
from sklearn.linear_model import LinearRegression
from utils import load_data, split_data, evaluate_model

def main():

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)


    r2, mse = evaluate_model(model, X_test, y_test)
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    joblib.dump(model, "model.joblib")
    print("Model saved as model.joblib")

if __name__ == "__main__":
    main()
