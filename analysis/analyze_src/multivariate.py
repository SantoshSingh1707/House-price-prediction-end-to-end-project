from abc import ABC , abstractmethod

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class MultivariateAnalysisTemplate(ABC):
    def analyse(self,df:pd.DataFrame):
        self.generate_heatmap(df)
        self.generate_pairplot(df)
        
    
    @abstractmethod 
    def generate_heatmap(self,df:pd.DataFrame):
        pass
    
    @abstractmethod
    def generate_pairplot(self,df:pd.DataFrame):
        pass

class MultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(12,10))
        sns.heatmap(df.corr(),annot=True,fmt=".2f" , cmap="coolwarm" , linewidths=0.5)
        plt.title("Heatmap Correlation")
        plt.show()
    
    def generate_pairplot(self, df: pd.DataFrame):
        plt.figure(figsize=(12,10))
        sns.pairplot(df)
        plt.suptitle("Pair polot of Selected Features ",y=1.02)
        plt.show()

if __name__=="__init__":
    pass