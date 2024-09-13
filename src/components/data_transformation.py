# Any code rlated to transformation
# How to convert castegorical featuers into neumerical
# How to handle One Hot Encoding
# How to handle labele encoding

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from ...src.exception import CustomException
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DataTransformationconfig = DataTransformationConfig()

    def get_data_transformer_object(self):
        # This function is responsible for data transformationbased on different types of data.
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), # Responsible for handling the missing values
                    ("scaler", StandardScaler()) # Doing the standard scaling
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), # trying to replace all the missing values with respect to mode
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor =ColumnTransformer(
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)