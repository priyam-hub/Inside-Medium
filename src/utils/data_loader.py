# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from logger.logger import LoggerSetup

data_loader_logger = LoggerSetup(logger_name = "data_loader.py", log_filename_prefix = "data_loader").get_logger()

class DataLoader:
    """
    A utility class for loading and saving CSV data using pandas DataFrame.

    This class helps manage reading from and writing to CSV files in a structured
    and reusable way, which is especially useful in data pipelines and ML workflows.

    Attributes:
    
        df                 {pd.DataFrame}           : The loaded DataFrame. Initially set to None and updated after loading.
    
    """

    def __init__(self) -> None:
        """
        Initialize the DataLoader instance with an empty DataFrame.
        
        """
        try:
            
            data_loader_logger.info("DataLoader initialized successfully.")

        except Exception as e:
            data_loader_logger.error(f"Error initializing DataLoader: {repr(e)}")
            
            raise e

    def data_loader(self, file_path : str) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.

        Arguments:
        
            file_path              {str}           : The path to the CSV file to be read.

        Returns:
        
            pd.DataFrame                           : The loaded DataFrame containing the CSV data.

        """
        try:
       
            self.df = pd.read_csv(file_path)
            data_loader_logger.info(f"Data loaded successfully from {file_path}.")
       
            return self.df
       
        except Exception as e:
            data_loader_logger.error(f"Error loading data from {file_path}: {repr(e)}")
            
            raise e

    def data_saver(self, dataframe : pd.DataFrame, file_path : str) -> None:
        """
        Save the current DataFrame to a CSV file.

        Arguments:
        
            file_path                {str}                 : The path where the CSV file will be saved.

        Returns:
        
            None

        """
        if dataframe is None:

            data_loader_logger.error("No DataFrame available to save. Load data first using `data_loader`.")
            raise ValueError("No DataFrame available to save. Load data first using `data_loader`.")
        
        try:

            dataframe.to_csv(file_path, index=False)
            data_loader_logger.info(f"Data saved successfully to {file_path}.")

        except Exception as e:
            data_loader_logger.error(f"Error saving data to {file_path}: {repr(e)}")
            
            raise e