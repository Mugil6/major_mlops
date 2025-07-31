import joblib
from utils import load_data, split_data

def main():
    model = joblib.load("model.joblib")
    X, y = load_data()
    _, X_test, _, _ = split_data(X, y)
    preds = model.predict(X_test)
    print("Sample predictions:", preds[:5])

if __name__ == "__main__":
    main()
