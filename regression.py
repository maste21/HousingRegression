import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data, split_data, train_models, evaluate_models

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    for name, metrics in results.items():
        print(f"{name} => MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.4f}")

if __name__ == "__main__":
    main()