from abc import ABC , abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class UnivariateAnalysisStratergy(ABC):
    @abstractmethod
    def analyze(self , df:pd.DataFrame , feature:str):
        pass

class NumericalUnivariateAlaysis(UnivariateAnalysisStratergy):
    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10,6))
        sns.histplot(x=df[feature],kde=True,bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("frequency")
        plt.show()

class CategoricalUnivariateAnalysis(UnivariateAnalysisStratergy):
    def analyze(self, df: pd.DataFrame, feature: str):
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature,data=df,palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

class UnivariateAnalyzer:
    def __init__(self , strategy:UnivariateAnalysisStratergy):
        self.strategy = strategy
    
    def set_strategy(self,strategy:UnivariateAnalysisStratergy):
        self.strategy = strategy
        
    def execute_analysis(self,df:pd.DataFrame , feature:str):
        self.strategy.analyze(df,feature)

if __name__=="__main__":
    pass