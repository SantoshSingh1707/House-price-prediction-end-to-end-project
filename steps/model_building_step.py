import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig,step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="price_predictor" , 
    version=None,
    license="Apache 2.0",
    description="Pirce prediction model for houses",
    )

@step(enable_cache=False , experiment_tracker=experiment_tracker.name , model=model)
def model_building_step(X_train:pd.DataFrame , y_train:pd.Series)->Annotated[Pipeline,ArtifactConfig(name="sklearn_pipeline" ,is_model_artifact=True)]:
    if not isinstance(X_train,pd.DataFrame):
        raise ValueError("X_train must be pandas DataFrame")
    if not isinstance(y_train,pd.Series):
        raise ValueError("y_train must be pandas Series")
    
    categorical_clos = X_train.select_dtypes(include=["object","category"]).columns
    numerical_clos = X_train.select_dtypes(exclude=["object","category"]).columns

    logging.info(f"Categorical feature = {categorical_clos}")
    logging.info(f"Numerical feature = {numerical_clos}")

    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot" , OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("nums",numerical_transformer,numerical_clos),
            ("cat" , categorical_transformer,categorical_clos)
        ]
    )
    pipeline = Pipeline(steps=[("preprocesser",preprocessor),("model",LinearRegression())])

    if not mlflow.active_run():
        mlflow.start_run()

    try:
        mlflow.sklearn.autolog()

        logging.info("Building and Training the Liner Regression model")
        pipeline.fit(X_train,y_train)
        logging.info("Model Training completed")

        onehot_encoder = (pipeline.named_steps["preprocesser"].transformers_[1][1].named_steps["onehot"])
        onehot_encoder.fit(X_train[categorical_clos])
        expected_columns = numerical_clos.tolist()+list(onehot_encoder.get_feature_names_out(categorical_clos))
        logging.info(f"Model expects the following columns :{expected_columns}")

    except Exception as e:
        logging.info(f"Error during Model training {e}")
        raise e

    finally:
        mlflow.end_run()
    
    return pipeline