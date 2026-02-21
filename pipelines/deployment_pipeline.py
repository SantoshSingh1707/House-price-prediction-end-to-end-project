import os 

from pipelines.training_pipeline import ml_pipeline
from zenml import pipeline
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

@pipeline
def continuous_deploymnet_pipeline():
    trained_model = ml_pipeline()

    
    
    mlflow_model_deployer_step(
    workers=3,
    deploy_decision=True,
    mlserver=False,
    model=trained_model
)

@pipeline(enable_cache=False)
def inference_pipeline():
    batch_data = dynamic_importer()

    model_deplotment_service = prediction_service_loader(
        pipeline_name ="continuous_deploymnet_pipeline",
        step_name="mlflow_model_deployer_step",
    )
    
    predictor(service=model_deplotment_service,input_data=batch_data)
