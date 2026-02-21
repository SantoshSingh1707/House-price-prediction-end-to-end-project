import os
import zipfile
from abc import ABC,abstractmethod
import pandas as pd

class DataIngestor(ABC):
    @abstractmethod
    def ingest(self,filepath: str)-> pd.DataFrame:
        pass

class ZipDataIngestor(DataIngestor):
    def ingest(self,filepath:str)->pd.DataFrame:
        if not filepath.endswith(".zip"):
            raise ValueError("The Provided file is not a .zip file")
        
        with zipfile.ZipFile(filepath,'r') as zip_ref:
            zip_ref.extractall("extracted_data")
        
        extracted_data = os.listdir("extracted_data")
        csv_file = [f for f in extracted_data if f.endswith(".csv")]

        if len(csv_file) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_file)>1:
            raise ValueError("Multiple CSV file found.Please specify which to use.")
        
        csv_file_path = os.path.join("extracted_data",csv_file[0])
        df = pd.read_csv(csv_file_path)

        return df
    
class DataIngestorFacorty:
    @staticmethod
    def get_data_ingestor(file_extension:str)->DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension:{file_extension}")
        
if __name__=="__main__":
    
    pass
