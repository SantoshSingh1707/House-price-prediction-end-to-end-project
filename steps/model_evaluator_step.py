import logging
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator,RegressionModelEvaluationStrategy
from zenml import step

@step(enable_cache=False)
def model_evaluator_step(
    trained_model:Pipeline,X_test:pd.DataFrame,y_test:pd.Series
)->Tuple[dict,float]:
    
    if not isinstance(X_test,pd.DataFrame):
        raise ValueError ("X_test must be pandas Dataframe")
    if not isinstance(y_test,pd.Series):
        raise ValueError("y_test must be pandas series")
    
    logging.info("Applying the same preprocessing to the test data")

    X_test_processed = trained_model.named_steps["preprocesser"].transform(X_test)

    evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())

    evaluation_metrics = evaluator.evaluate(
        trained_model.named_steps["model"], X_test_processed, y_test
    )
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as dict")

    mse = evaluation_metrics.get("Mean Squared error", None)

    return evaluation_metrics, mse
