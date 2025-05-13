# DEPENDENCIES

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.data_loader import DataLoader
from src.utils.download_dataset import DownloadData
from src.data_preprocessor.data_preprocessor import DataPreprocessor
from src.exploratory_data_analysis.exploratory_data_analyzer import MediumEDA

import warnings
warnings.filterwarnings(action = "ignore")

# LOGGER SETUP
main_logger = LoggerSetup(logger_name = "main.py", log_filename_prefix = "main").get_logger()

def main():

    try:

        # Save this in your script
        downloader            = DownloadData(dataset_name   = "dorianlazar/medium-articles-dataset",
                                             download_path  = "./data/"
                                             )

        downloader.download_dataset()

        main_logger.info("Dataset downloaded successfully.")

        dataLoader            = DataLoader()   
        medium_raw_df         = dataLoader.data_loader(file_path = Config.MEDIUM_RAW_DATASET_PATH)
        main_logger.info("Data loaded successfully:")

        # data_analyzer         = MediumEDA(dataframe       = medium_raw_df, 
        #                                   output_dir      = Config.EDA_RESULTS_PATH)    
         
        # data_analyzer.run_all_eda()
        # main_logger.info("All EDA completed successfully.")

        preprocessor          = DataPreprocessor(dataFrame = medium_raw_df)
        medium_processed_df   = preprocessor.preprocess_data()

        main_logger.info("Data preprocessing completed successfully.")

        dataLoader.data_saver(dataframe = medium_processed_df, 
                              file_path = Config.MEDIUM_PROCESSED_DATASET_PATH
                              )
        
        main_logger.info("Processed data saved successfully.")

    except Exception as e:
        
        print(f"Error Occurred In PipeLine: {repr(e)}")
        return


if __name__ == "__main__":
    main()