# DEPENDENCIES

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.data_loader import DataLoader
from src.exploratory_data_analysis.exploratory_data_analyzer import MediumEDA

# LOGGER SETUP
main_logger = LoggerSetup(logger_name = "main.py", log_filename_prefix = "main").get_logger()

def main():

    try:

        dataLoader            = DataLoader()   
        medium_raw_df         = dataLoader.data_loader(file_path = Config.MEDIUM_RAW_DATASET_PATH)
        main_logger.info("Data loaded successfully:")

        data_analyzer         = MediumEDA(dataframe       = medium_raw_df, 
                                          output_dir      = Config.EDA_RESULTS_PATH)    
         
        data_analyzer.run_all_eda()
        main_logger.info("All EDA completed successfully.")


    except Exception as e:
        
        print(f"Error Occurred In PipeLine: {repr(e)}")
        return


if __name__ == "__main__":
    main()