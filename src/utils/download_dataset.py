# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import zipfile
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from logger.logger import LoggerSetup

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# SET ENV VARIABLES FOR KAGGLE API
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_API_KEY"]  = os.getenv("KAGGLE_API_KEY")

# LOGGER SETUP
data_downloader_logger        = LoggerSetup(logger_name = "download_dataset.py", log_filename_prefix = "download_dataset").get_logger()


class DownloadData:
    """
    A class to handle downloading datasets from Kaggle.

    This class uses the Kaggle API to authenticate and download the full Medium articles dataset
    from the specified dataset URL or identifier.

    Attributes:
    
        `dataset_name`                  {str}       : The Kaggle dataset identifier in 'username/dataset-name' format.
        
        `download_path`                 {str}       : The local path where the dataset should be downloaded and extracted.
    
    """

    def __init__(self, dataset_name : str, download_path : str) -> None:
        """
        Initialize the DownloadData object.

        Arguments:

            `dataset_name`              {str}      : The Kaggle dataset slug (e.g., 'dorianlazar/medium-articles-dataset').
            
            `download_path`             {str}      : The path where the dataset will be downloaded and extracted.
        
        """
        
        try:
            
            if not isinstance(dataset_name, str):
                data_downloader_logger.error("[DownloadData] Dataset name must be a string.")
                
                raise ValueError("Dataset name must be a string.")
            
            if not isinstance(download_path, str):
                data_downloader_logger.error("[DownloadData] Download path must be a string.")
                
                raise ValueError("Download path must be a string.")
            
            self.dataset_name   = dataset_name
            self.download_path  = download_path
            self.api            = KaggleApi()
            self.api.authenticate()

            data_downloader_logger.info(f"[DownloadData] Initialized with dataset: {self.dataset_name} and download path: {self.download_path}")

        except Exception as e:
            data_downloader_logger.error(f"[DownloadData] Error initializing DownloadData: {repr(e)}")
            
            raise

    def download_dataset(self) -> None:
        """
        Download the dataset from Kaggle and extract it to the specified folder.

        Downloads the entire dataset ZIP file and extracts it in the provided `download_path`.

        Raises:
            
            Exception: If any error occurs during download or extraction.
        
        """
        
        try:
        
            if not os.path.exists(self.download_path):
                os.makedirs(self.download_path)

            data_downloader_logger.info(f"Downloading dataset: {self.dataset_name}")
            self.api.dataset_download_files(self.dataset_name, path = self.download_path, unzip = True)

            data_downloader_logger.info(f"Dataset downloaded and extracted successfully to: {self.download_path}")

        except Exception as e:
            data_downloader_logger.error(f"Failed to download or extract dataset: {repr(e)}")
            
            raise
