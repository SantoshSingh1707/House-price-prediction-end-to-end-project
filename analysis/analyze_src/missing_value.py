from abc import ABC , abstractmethod

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class MissingValuesAnalysisTemplate(ABC):
    def analyse(self , df:pd.DataFrame):
        self.identify_missing_value(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_value(self,df:pd.DataFrame):
        pass

    @abstractmethod
    def visualize_missing_values(self,df:pd.DataFrame):
        pass


class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_value(self, df: pd.DataFrame):
        print("Missing Values Count by columns :")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values>0])
    
    def visualize_missing_values(self, df: pd.DataFrame):
        print("Visualizing Missing Values :")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
        plt.title("Missing Values HeatMap")
        plt.show()

if __name__=="__main__":
    pass