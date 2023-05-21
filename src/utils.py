from src.logger import logging
from src.exception import CustomException
import pymongo
from pymongo import MongoClient
import pandas as pd
import sys
import os
import certifi
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error



def MongoDBcoll():
    logging.info("Connecting to MongoDB database")
    try:
        ca= certifi.where()
        cluster = MongoClient("mongodb+srv://arup92327:Arup0070@cluster0.e9r83iz.mongodb.net/?retryWrites=true&w=majority",tlsCAFile=ca)
        db=cluster["ZomatoDB"]
        coll=db["Zomato_Data"]
        li=[]
        for i in coll.find():
            li.append(i)
        Data=pd.DataFrame(li)
        logging.info(f"{Data.head()} \n Collected Data")
        return Data
    except Exception as e:
        logging.info("not able to collect data from MongoBD Data Base")
        raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
        logging.info("Saveing Pickel obj in file")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info("error in Object Save module")
        raise CustomException(e,sys)


def evaluate_model(x_train,y_train,x_test,y_test,models):

    try:
        report = {}
        for i in range(len(models)):
            model=list(models.values())[i]
            model.fit(x_train,y_train)
    
            y_pred=model.predict(x_test)

            test_model_score = r2_score(y_test,y_pred)
            report[list(models.keys())[i]] = test_model_score

            return report
    except Exception as e:
        logging.info("Error occured at model evaluate module")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(os.getcwd()+"\\"+file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("error occured at model load module")
        raise CustomException(e,sys)
    
