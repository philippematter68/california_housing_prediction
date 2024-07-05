#load dependencies
import os
import dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.pipeline import Pipeline
from src.data.make_dataset import main as make_dataset
from src.data.make_dataset import read_train_test_data

# Load data
project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, '.env')
dotenv.load_dotenv(dotenv_path)

data_path = os.getenv("DATA_PATH")
processed_path = os.getenv("PROCESSED_PATH")
make_dataset(data_path, processed_path)

X_train, X_test, y_train, y_test = read_train_test_data(processed_path)

#  Modelling
#  Pipeline Definition

sc = StandardScaler()
lin_reg = LinearRegression()
pipeline_mlr = Pipeline([("data_scaling", sc), ("estimator", lin_reg)])

# ### Model Fit

pipeline_mlr.fit(X_train, y_train)

# ## [Model Evaluation](#P0)

predictions_mlr = pipeline_mlr.predict(X_test)

# Test score
pipeline_mlr.score(X_test, y_test)

print("MAE", metrics.mean_absolute_error(y_test, predictions_mlr))
print("MSE", metrics.mean_squared_error(y_test, predictions_mlr))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, predictions_mlr)))
print("Explained Var Score", metrics.explained_variance_score(y_test, predictions_mlr))




