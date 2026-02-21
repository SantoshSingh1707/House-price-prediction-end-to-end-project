import pandas as pd
from src.handel_missing_value import DropMissingValueStrategy,FillMissingValuesStrategy,MissingValueHandler

from zenml import step

@step
def handle_missing_values_step(df:pd.DataFrame,strategy:str="mean"):
    if strategy=="drop":
        handler = MissingValueHandler(DropMissingValueStrategy(axis=0))
    elif strategy in ["mean" , "meadian" ,"mode" , "constant"]:
        handler = MissingValueHandler(FillMissingValuesStrategy(method=strategy))
    else:
        raise ValueError("Unsupported missing values handling strategy")
    
    df_cleaned = handler.handle_missing_values(df)
    return df_cleaned