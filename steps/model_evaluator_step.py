import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator,RegressionModelEvaluationStrategy
from zenml import step

@step(enable_cache=False)
def model_evalautor_step(
    trained_model:Pipeline,X_test:pd.DataFrame,y_test:pd.Series
)->Tuple[dict,float]:
    
    if not isinstance(X_test,pd.DataFrame):
        raise ValueError ("X_test must be pandas Dataframe")
    if not isinstance(y_test,pd.Series):
        raise ValueError("y_test must be pandas series")
    
    logging.info("Appling the same preprocssing to the test data")

    X_test_processd = trained_model.named_steps["preprocesser"].transform(X_test)

    evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())

    evalaution_metrics = evaluator.evaluate(
        trained_model.named_steps["model"],X_test_processd,y_test
    )
    if not isinstance(evalaution_metrics,dict):
        raise ValueError("Evalaution metrics must be retruned as dict")

    mse = evalaution_metrics.get("Mean Squared error",None)

    return evalaution_metrics,mse
