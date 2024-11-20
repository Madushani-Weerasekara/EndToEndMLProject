# Utils will have the all the common things that we are going to import or use
# Which will be used in entire application
# Read a dataset from database I can create a mongo db client here
# If I want to save the model into the cloud I can write the code here

import os
import sys

import numpy as np
import pandas as pd
import dill # Library that will help to create the pickle file
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def evaluate_models(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():

            model = list(models.values())[model_name]
            para=param[list(models.keys())[model_name]]

            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            try:

                model.fit(x_train, y_train) # Train model
                 
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = r2_score(y_train, y_train_pred)

                test_model_score = r2_score(y_test, y_test_pred)
            except Exception as e:
                print(f"Model {model_name} failed with error {e}")

             
 
             
        return report
    
    except:
        pass

def save_object(file_path, obj):
    try:
        dir_path =os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)