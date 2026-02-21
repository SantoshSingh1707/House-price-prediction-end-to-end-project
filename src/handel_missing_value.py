import logging
from abc import ABC,abstractmethod

import pandas as pd

logging.basicConfig(level=logging.INFO , format="%(asctime)s - %(levelname)s -%(message)s")

class MissingValueHandelingStrategy(ABC):
    @abstractmethod
    def handle(self, df:pd.DataFrame)->pd.DataFrame:
        pass

class DropMissingValueStrategy(MissingValueHandelingStrategy):
    def __init__(self,axis=0,thresh=None):
        self.axis = axis
        self.thresh = thresh
    
    def handle(self,df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Dropping Missing Values with axis={self.axis}")
        df_cleaned = df.dropna(axis=self.axis , thresh=self.thresh)
        logging.info("Missing values dropped ")
        return df_cleaned

class FillMissingValuesStrategy(MissingValueHandelingStrategy):
    def __init__(self,method="mean" , fill_value=None) :
        self.method = method
        self.fill_value = fill_value

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Filling missing values using method={self.method}")
        df_cleaned = df.copy()

        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method '{self.method}'. No missing values handled.")
        
        logging.info("Missing Value filled")
        return df_cleaned

class MissingValueHandler:
    def __init__(self,strategy:MissingValueHandelingStrategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:MissingValueHandelingStrategy):
        
        logging.info("Switching Missing Value handeling Strategy.")

        self.strategy = strategy
    
    def handle_missing_values(self , df:pd.DataFrame)->pd.DataFrame:

        logging.info("Executing missing value handeling strategy.")
        return self.strategy.handle(df)
    
if __name__=="__main__":
    pass