import logging
from abc import ABC , abstractmethod
from typing import Any

import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,X_train:pd.DataFrame,y_train:pd.Series)->RegressorMixin:
        pass

class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train: pd.DataFrame, y_train: pd.Series[Any]) -> RegressorMixin:
        if not isinstance(X_train,pd.DataFrame):
            raise ValueError("X_train must be pandas dataframe")
        if not isinstance(y_train,pd.Series):
            raise ValueError("y_train must be pandas series")
        
        logging.info("Initializing Linear regression model with Scaling")

        pipeline = Pipeline([
            ("scaler",StandardScaler()),
            ("model",LinearRegression()),
        ])
        logging.info("Training Linear Regression model")
        pipeline.fit(X_train,y_train)

        logging.info("Model Trainig Completed")
        return pipeline

class ModelBuilder:
    def __init__(self,strategy:ModelBuildingStrategy):
        self.strategy = strategy
    
    def set_startegy(self,strategy:ModelBuildingStrategy):
        logging.info("Switching model building strategy")
        self.strategy = strategy
    
    def build_model(self,X_train:pd.DataFrame,y_train:pd.Series)->RegressorMixin:
        logging.info("Building and training the model using Selected strategy")
        return self.strategy.build_and_train_model(X_train,y_train)

    if __name__=="__main__":
        pass