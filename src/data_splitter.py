import logging
from abc import ABC , abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataSplitterStrategy(ABC):
    @abstractmethod
    def split_data(self,df:pd.DataFrame,target_column:str):
        pass

class SimpleDataSplitStrategy(DataSplitterStrategy):
    def __init__(self,test_size = 0.2,random_state = 42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self,df:pd.DataFrame,target_columns:str):
        logging.info("Performing simple train-test split")
        X = df.drop(columns=[target_columns])
        y = df[target_columns]

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state)
        return X_train,X_test,y_train,y_test

class DataSplitter:
    def __init__(self,strategy:DataSplitterStrategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:DataSplitterStrategy):
        self.strategy = strategy
    
    def split(self,df:pd.DataFrame,target_column:str):
        logging.info("Spliting data using selected strategy")
        return self.strategy.split_data(df,target_column)
    

if __name__=="__main__":
    pass
        