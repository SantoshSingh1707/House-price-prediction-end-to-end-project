import logging
from abc import ABC,abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self,df:pd.DataFrame)->pd.DataFrame:
        pass


class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features = features
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log tranformation to features :{self.features}")
        df_transformed = df.copy()

        for features in self.features:
            df_transformed[features] = np.log1p(df[features])

        logging.info("Log tranformation completed")
        return df_transformed
    
class StanderdScaling(FeatureEngineeringStrategy):
    def __init__(self,features):

        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Standard scaling to features : {self.features}")
        df_scaled = df.copy()
        df_scaled[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standred Scaler completed")
        return df_scaled

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,features,features_range=(0,1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=features_range)
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Appling Min-Max scaling to features :{self.features} ")
        df_tranformed = df.copy()
        df_tranformed[self.features] = self.scaler.fit_transform(df_tranformed[self.features])
        logging.info("Min-Max scaler completed")
        return df_tranformed

class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False,drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying One-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns = self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed,encoded_df])
        logging.info("One-hot encoding completed")
        return df_transformed

class Feature_Engineering:
    def __init__(self,strategy:FeatureEngineeringStrategy):
        self.strategy = strategy
    
    def set_startegy(self,strategy:FeatureEngineeringStrategy):
        logging.info("Switching feature engineering strategy")
        self.strategy = strategy
    
    def apply_feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        logging.info("Applying feature engineering startegy")
        return self.strategy.apply_transformation(df)
    

if __name__=="__main__":
    pass