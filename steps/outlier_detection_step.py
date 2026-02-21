import logging

import pandas as pd
from src.outlier_detection import OutliersDetector,ZScoreOutlierDetection
from zenml import step

@step
def outlier_detection_step(df:pd.DataFrame,columns_name:str)->pd.DataFrame:
    logging.info(f"Strating outlier detection step with Dataframe of shape {df.shape}")

    if df is None:
        logging.error("Received a None type dataset")
        raise ValueError("Input dataset must a non-null pandas dataframe")
    if not isinstance(df,pd.DataFrame):
        logging.error(f"Expected pandas Dataframe , got {type(df)} instead")
        raise ValueError("Input df must be pandas Dataframe")
    
    if columns_name not in df.columns:
        logging.error(f"column {columns_name} does not exist in dataset")
        raise ValueError(f"Column {columns_name} does not exist in dataset")
    df_numeric = df.select_dtypes(include=[int,float])

    outlier_detector = OutliersDetector(ZScoreOutlierDetection(threshold=3))
    outliers = outlier_detector.detect_outlier(df_numeric)
    df_cleaned = outlier_detector.handle_outlier(df_numeric,method="remove")
    return df_cleaned
