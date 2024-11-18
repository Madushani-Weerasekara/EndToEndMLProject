# Will have all the code reladed to reading the data
# devide the dataset into train and test
# create a validation data
 
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    #These are the inputs that giving to data_ingestion component
    train_data_path: str=os.path.join('artifacts', "train.csv")
    test_data_path: str=os.path.join('artifacts', "test.csv")
    raw_data_path: str=os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig(

        )
        
    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")
        
        try:
            data_file_path = os.path.join(os.getcwd(), 'src','notebook', 'data', 'exams_performance.csv')
            df = pd.read_csv(data_file_path)

            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    # Combine data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion() # when done obj.initiate_data_ingestion() will return two values(train_data and test_data)

    # Combine data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path))





