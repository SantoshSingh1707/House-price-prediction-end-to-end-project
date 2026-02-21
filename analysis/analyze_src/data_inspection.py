from abc import ABC ,abstractmethod

import pandas as pd

class DataInspectionStratergy(ABC):
    @abstractmethod
    def inspect(self,df:pd.DataFrame):
        pass


class DataTypeInspectionStrategy(DataInspectionStratergy):
    def inspect(self,df:pd.DataFrame):

        print("Data Types and null Counts :")
        print(df.info())


class SummaryStatisticsInspectionStrategy(DataInspectionStratergy):
    def inspect(self, df: pd.DataFrame):
        print("Summary Statistics (Numerical Features) :")
        print(df.describe())
        print("Summary Statistics (Categorical Features) :")
        print(df.describe(include=['O']))

class DataInspector:
    def __init__(self,strategy:DataInspectionStratergy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:DataInspectionStratergy):
        self.strategy = strategy

    def execute_inspection(self,df:pd.DataFrame):
        self.strategy.inspect(df)

if __name__ == "__main__":

    pass