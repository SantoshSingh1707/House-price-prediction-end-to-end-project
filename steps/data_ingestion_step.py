import pandas as pd
from src.ingest_data import DataIngestorFacorty
from zenml import step

@step
def data_ingestion_step(file_path:str):
    file_extension = ".zip"

    data_ingestor = DataIngestorFacorty.get_data_ingestor(file_extension)

    df = data_ingestor.ingest(file_path)

    return df