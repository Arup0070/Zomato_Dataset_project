import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass
import pandas as pd

import numpy as np
from sklearn.impute import SimpleImputer#handeling Missing Values
from sklearn.preprocessing import StandardScaler#Featuer Scallling
from sklearn.preprocessing import OrdinalEncoder#Feature Encoading
#For pipelne
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationcoonfig:
    preprosessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformaion:
    def __init__(self) :
        self.data_transformation_config = DataTransformationcoonfig()

    def get_data_traformation_object(self):
        try:
            logging.info("Data transformaton initiated")
            catagorical_col=['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle','Festival', 'City']
            numerical_col=['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries']

            Weather_cat=["Sunny","Stormy","Sandstorms","Windy","Fog","Cloudy"]
            Road_cat=["Low","Medium","High","Jam"]
            vehicle_cat = ["motorcycle","electric_scooter","scooter","bicycle",]
            Festival_cat=["No","Yes"]
            City_cat=["Urban","Metropolitian","Semi-Urban"]

            #Numerical Pipeline
            num_pipeline=Pipeline(
                                 steps=[('imputer',SimpleImputer(strategy="median")),
                                        ('scaler',StandardScaler())
                                        ]
                                )
            cat_pipeline=Pipeline(
                         steps=[ ('imputer',SimpleImputer(strategy='most_frequent')),
                                 ('ordinalencoder',OrdinalEncoder(categories=[Weather_cat,Road_cat,vehicle_cat,Festival_cat,City_cat])),
                                 ('scaler',StandardScaler())
                                ]
                                )
            preprossor=ColumnTransformer([
                                 ('num_pipeline',num_pipeline,numerical_col),
                                 ('cat_pipeline',cat_pipeline,catagorical_col)
                                 ]
                                 )
            return preprossor
            logging.info("Preprosessor OBJ created")

        except Exception as e:
            logging.info("Error occured at Data transformation stage")
            raise CustomException(e,sys)
        

    def initaite_data_transformation(self,train_path,test_path):

        try:
            logging.info("Data Transformation Process iniciated")

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info(" Train and test Data Read Complete")

            preprocessing_obj=self.get_data_traformation_object()

            target_column = 'Time_taken_(min)'
            drop_columns = [target_column,'_id']

            input_features_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column]

            input_features_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]


            #Transformation useing preprocessor obj

            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            logging.info("Applying preprocessing object on training and testing")

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprosessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprosessor_obj_file_path
            )

            
        except Exception as e:
            logging.info("Error in Data transformation Module")
            raise CustomException(e,sys)
        