from abc import ABC , abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyse(self,df:pd.DataFrame , feature1:str , feature2:str):
        pass

class NumericalvsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    
class NumericalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyse(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()
    
class BivaraiteAnalyzer:
    def __init__(self,strategy:BivariateAnalysisStrategy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:BivariateAnalysisStrategy):
        self.strategy = strategy
    
    def execute_analysis(self,df:pd.DataFrame , feature1:str , feature2:str):
        self.strategy.analyse(df,feature1,feature2)
    

if __name__=="__init__":
    pass