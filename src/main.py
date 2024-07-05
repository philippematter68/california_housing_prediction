#load dependencies
import os
import dotenv
from dill import dump, load
import logging
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

# defines the folder for the model, if it does not exists already.
model_path = os.getenv("MODEL_PATH")
if not os.path.exists(model_path):
    os.makdir(model_path)
model_file_name = model_path + "/LinearRegression.pkl"

make_dataset(data_path, processed_path)

X_train, X_test, y_train, y_test = read_train_test_data(processed_path)

#  Modelling:  Pipeline Definition
sc = StandardScaler()
lin_reg = LinearRegression()
pipeline_mlr = Pipeline([("data_scaling", sc), ("estimator", lin_reg)])
# Model Fit
pipeline_mlr.fit(X_train, y_train)
logger.debug(f"Saving model as: {model_file_name}")
with open(model_file_name, "wb") as f:
    dump(pipeline_mlr, f)
# Model Evaluation
with open(model_file_name, "rb") as f:
    pipeline_mlr = load(f)
predictions_mlr = pipeline_mlr.predict(X_test)

# Test score
pipeline_mlr.score(X_test, y_test)

logger.info(f"MAE {metrics.mean_absolute_error(y_test, predictions_mlr)}")
logger.info(f"MSE {metrics.mean_squared_error(y_test, predictions_mlr)}")
logger.info(f"RMSE {np.sqrt(metrics.mean_squared_error(y_test, predictions_mlr))}")
logger.info(f"Explained Var Score {metrics.explained_variance_score(y_test, predictions_mlr)}")




