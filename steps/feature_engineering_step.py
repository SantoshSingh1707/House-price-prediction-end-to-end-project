import pandas as pd

from src.feature_engineering import Feature_Engineering , MinMaxScaling,StanderdScaling,LogTransformation,OneHotEncoding

from zenml import step

@step
def feature_engineering_step(df:pd.DataFrame , strategy:str="log" , features:list = None )->pd.DataFrame:
    if features == None:
        features=[]
    
    if strategy=="log":
        engineer = Feature_Engineering(LogTransformation(features))
    elif strategy=="standard_scaling":
        engineer = Feature_Engineering(StanderdScaling(df))
    elif strategy=="minmax_scaling":
        engineer = Feature_Engineering(MinMaxScaling(df))
    elif strategy=="onehot_encoding":
        engineer=Feature_Engineering(OneHotEncoding(df))
    else:
        raise ValueError(f"Unsupported feature engineering strategy : {strategy}")
    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df

