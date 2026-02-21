from typing import Tuple

import pandas as pd
from src.data_splitter import DataSplitter , SimpleDataSplitStrategy
from zenml import step

@step
def data_splitter_step(df:pd.DataFrame,target_columns:str)->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    splitter = DataSplitter(SimpleDataSplitStrategy())
    X_train,X_test,y_train,y_test = splitter.split(df,target_columns)
    return X_train,X_test,y_train,y_test
