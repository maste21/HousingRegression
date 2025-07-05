{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c2f8f774-61a1-40a2-908f-1e985557f435",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "def load_data():\n",
    "    data_url = \"http://lib.stat.cmu.edu/datasets/boston\"\n",
    "    raw_df = pd.read_csv(data_url, sep=r\"\\s+\", skiprows=22, header=None)\n",
    "    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])\n",
    "    target = raw_df.values[1::2, 2]\n",
    "    feature_names = [\n",
    "        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',\n",
    "        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'\n",
    "    ]\n",
    "    df = pd.DataFrame(data, columns=feature_names)\n",
    "    df['MEDV'] = target\n",
    "    return df\n",
    "\n",
    "def split_data(df):\n",
    "    X = df.drop('MEDV', axis=1)\n",
    "    y = df['MEDV']\n",
    "    return train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "def train_models(X_train, y_train):\n",
    "    models = {\n",
    "        'LinearRegression': LinearRegression(),\n",
    "        'DecisionTree': DecisionTreeRegressor(random_state=42),\n",
    "        'RandomForest': RandomForestRegressor(random_state=42)\n",
    "    }\n",
    "    for model in models.values():\n",
    "        model.fit(X_train, y_train)\n",
    "    return models\n",
    "\n",
    "def evaluate_models(models, X_test, y_test):\n",
    "    results = {}\n",
    "    for name, model in models.items():\n",
    "        preds = model.predict(X_test)\n",
    "        mse = mean_squared_error(y_test, preds)\n",
    "        r2 = r2_score(y_test, preds)\n",
    "        results[name] = {'MSE': mse, 'R2': r2}\n",
    "    return results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0474534d-a38e-4d81-9a0d-ecb0bcc63004",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6f9a08ad-0376-4733-8718-f32e90de3c90",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
