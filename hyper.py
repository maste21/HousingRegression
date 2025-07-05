# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset function (include if not defined yet)
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

# Split function (include if not defined yet)
def split_data(df):
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluation function
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {'MSE': mse, 'R2': r2}
    return results

# Parameter grids for tuning
def get_param_grids():
    grids = {
        'LinearRegression': {},  # No hyperparameters for LinearRegression
        'DecisionTree': {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
        }
    }
    return grids

# GridSearch wrapper function
def perform_grid_search(model, param_grid, X_train, y_train):
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_
    else:
        model.fit(X_train, y_train)
        return model, {}

# Tune all models
def tune_all_models(X_train, y_train):
    base_models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42)
    }

    param_grids = get_param_grids()
    best_models = {}

    for name, model in base_models.items():
        print(f"Tuning {name}...")
        best_model, best_params = perform_grid_search(model, param_grids[name], X_train, y_train)
        if best_params:
            print(f"Best Params for {name}: {best_params}")
        best_models[name] = best_model

    return best_models

# Run entire hyper tuning pipeline
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
tuned_models = tune_all_models(X_train, y_train)
tuned_results = evaluate_models(tuned_models, X_test, y_test)

print("\nTuned Model Performance:\n")
for name, metrics in tuned_results.items():
    print(f"{name} => MSE: {metrics['MSE']:.2f}, R²: {metrics['R2']:.4f}")