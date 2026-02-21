import logging
from abc import ABC,abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error,r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(
        self,model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series
    )->dict:
        pass


class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        logging.info("Predictiong using the train model")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics")
        mse = mean_squared_error(y_test,y_pred)
        r2 = r2_score(y_test,y_pred)

        metrics = {"Mean Squared error":mse, "r2 Score":r2}
        logging.info(f"Model evalution metrics :{metrics}")
        return metrics
    
class ModelEvaluator:
    def __init__(self,strategy:ModelEvaluationStrategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:ModelEvaluationStrategy):
        logging.info("Switching model evaluation strategy")
        self.strategy = strategy
    
    def evaluate(self,model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series)->dict:
        logging.info("Evaluating model using Selected model")
        return self.strategy.evaluate_model(model,X_test,y_test)

if __name__=="__main__":
    pass