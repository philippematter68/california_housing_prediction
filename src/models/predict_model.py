### Production-ready predict function
from pydantic import BaseModel, field_validator, ValidationError
import pandas as pd
import pickle
import numpy as np

class PredictionInput(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    population: float
    households: float
    median_income: float

    @field_validator("longitude")
    def longitude_is_negative(cls, value):
        if value > 0:
             raise ValueError("longitude should be lower than 0")
        return value

def verify_input_with_pydantic_dm(input_data:pd.DataFrame, DataModel:BaseModel=PredictionInput)->pd.DataFrame:
    """uses a pydantic data model tocheck each row of a pndas dataframe with input data

    Args:
        input_data (pd.DataFrame): dataframe to check
        DataModel (BaseModel, optional): dataModel defining data expectations. Defaults to PredictionInput.

    Raises:
        ValueError: if datamodel validation raises an error

    Returns:
        pd.DataFrame: verified_data
    """
    verified_inputs=[]
    for n,i in input_data.iterrows():
        try:
            verified_input=DataModel(**dict(i))
            verified_inputs.append(verified_input.dict())
        except ValidationError as e:
            raise ValueError(f"input_data on line {n} does not confor to expectations: {e}")
    
    return pd.DataFrame(verified_inputs)
    
        
def robust_predict(model_path: str, input_data:pd.DataFrame, DataVerifier:callable=verify_input_with_pydantic_dm)->np.ndarray:
    """ 
    this functions outputs a prediction given a model and a sk_learn pipeline object as model.
    Created to decouple model building and prediciton and ensure conformity of input data

    Created to decouple model building and prediciton and ensure conformity of input data

    Args:
        model_path (str): path to trained_model
        input_data (pd.DataFrame): dataframe with input data for predictions
        dm (BaseModel): dataModel to che conformity of input data

    Returns:
        np.ndarray: output array of floats giving the predicted price of houses
    """
    with open(model_path, "rb") as f:
        model= pickle.load(f)
    
    verified_data=DataVerifier(input_data)

    return model.predict(verified_data)