# all the training code over here
# How many different kinds of model i want to use
# Probobly will call here the confusion Matrix if solving a clasification problem, R squard value if a regression problem

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn. tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer: # Responsible for model training
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(train_array, test_array, preprocessor_path):
        try:
            logging.info("Spliting training and test input data")
            x_train, x_test, y_train, y_test = (
                train_array[:, :-1], # Take out the last column & stor eeverythin in x_train
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Create a dictionary of models
            models ={
                "Random Frest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linesr Regression": LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoostingClassifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostRegressor()

            }

            model_report:dict = evaluate_models(x_train, y_train, x_test, y_test, models)

            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model in both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
            
        

