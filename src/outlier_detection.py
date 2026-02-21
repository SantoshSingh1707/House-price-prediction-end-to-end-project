import logging
from abc import ABC , abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OutlierDetecionStrategy(ABC):
    @abstractmethod
    def detect_outlier(self,df:pd.DataFrame)->pd.DataFrame:

        pass
    

class ZScoreOutlierDetection(OutlierDetecionStrategy):
    def __init__(self,threshold = 3) :
        self.threshold  = threshold
    
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting Outlier using Z-Score method")
        z_scores = np.abs((df-df.mean())/df.std())
        outlier = z_scores>self.threshold
        logging.info(f"Outliers With Z-score threshold :{self.threshold}")
        return outlier
    
class IQROutlierDetection(OutlierDetecionStrategy):
    def detect_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecing Outlier using IQR Method")
        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        IQR = q3-q1
        outlier = (df < (q1-1.5*IQR) | (df > (q3 + 1.5*IQR)))
        logging.info("Outliers detected using IQR method")
        return outlier

class OutliersDetector:
    def __init__(self , strategy:OutlierDetecionStrategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:OutlierDetecionStrategy):
        logging.info("Switching outlier detectio strategy")
        self.strategy = strategy

    def detect_outlier(self , df:pd.DataFrame)->pd.DataFrame:
        logging.info("Executing Outlier detection strategy")
        return self.strategy.detect_outlier(df)

    def handle_outlier(self,df:pd.DataFrame,method="remove",**kwargs)->pd.DataFrame:
        outliers = self.detect_outlier(df)
        if method == "remove":
            logging.info("Removing outliers from the dataset")
            df_cleaned = df[(~outliers).all(axis=1)]
        elif method=="cap":
            logging.info("Capping outliers in dataset")
            df_cleaned = df.clip(lower=df.quantile(0.01),upper=df.quantile(0.99) ,axis=1)
        else:
            logging.warning(f"Unknown method '{method}'.No outlier handeling performed")
            return df
        logging.info("Outlier handling completed")
        return df_cleaned
    
    def visialize_outliers(self,df:pd.DataFrame,features:list):
        logging.info(f"Visualizing outliers for features {features}")
        for feature in features:
            plt.figure(figsize=(10,6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Visualising outliers completed")
    
if __name__=="__main__":
    pass
