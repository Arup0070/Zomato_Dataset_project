
from sklearn.linear_model import LinearRegression , Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainingConfig:
    trained_model_file_path= os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_traner_config=ModelTrainingConfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("spliting dependent and independent verials from train and test data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={"LinearRegression":LinearRegression(),
                    "Ridge":Ridge(),
                    "Lasso":Lasso(),
                    "Elasticnet":ElasticNet(),
                    "RandomForestRegressor":RandomForestRegressor()
                   }
            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            print(model_report)
            print('\n=====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_traner_config.trained_model_file_path,
                 obj=best_model
            )

            
        except Exception as e:
            logging.info("error occured at model training module")
            raise CustomException(e,sys)
        