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


    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Spliting training and test input data")

            # Unpacking train and test data
            x_train, x_test, y_train, y_test = (
            train_array[:, :-1],  # Take out all columns except the last one for x_train
            test_array[:, :-1],   # Take out all columns except the last one for x_test
            train_array[:, -1],   # Use the last column as y_train
            test_array[:, -1]     # Use the last column as y_test
        
            )

    

            # Create a dictionary of models
            models ={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier" : KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoostingClassifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier" : AdaBoostRegressor()

            }

            params = {
                "Decision Tree": {
                    'criterion' : ['squared_error', ' friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter' : [''best', 'random'],
                    # 'max_featuures': ['sqrt', 'log2']
                },

                "Random Forest": {
                    # 'criterian': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterian': ['squared_error' , 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128,256]

                },

                "Linear Regression": {},

                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.01],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss': ['linear', 'square', 'exponential'],
                    'n_estimator': [8, 16, 32, 64, 128, 256]
                }
            }
                 

            # Evaluate models
            model_report:dict = evaluate_models(x_train, y_train, x_test, y_test, models, param=params)

             
            
            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

             # Check if the best model's score meets the threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model in both train and test dataset")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info(f"Saved the best model: {best_model_name} to {self.model_trainer_config.trained_model_file_path}")
             
            # Predict on test data and calculate R² score
            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
            
 
