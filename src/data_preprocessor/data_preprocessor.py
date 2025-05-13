# DEPENDENCIES

import re
import pandas as pd

from ..utils.logger import LoggerSetup

# LOGGER SETUP
preprocessor_logger = LoggerSetup(logger_name = "data_preprocessor.py", log_filename_prefix = "data_preprocessor").get_logger()

class DataPreprocessor:
    """
    A class used for preprocessing raw article data in a DataFrame.

    This class handles data cleaning tasks such as:
        - Converting date columns to datetime.
        - Removing HTML tags from titles.
        - Filling missing subtitles.
        - Creating a combined article field.
        - Dropping unnecessary columns.
        - Sorting the data based on number of claps.

    Attributes:
        
        df              {pd.DataFrame}          : The DataFrame containing raw data.
    
    """

    def __init__(self, dataFrame : pd.DataFrame) -> None:
        """
        Initialize the DataPreprocessor object.

        Arguments:

            df          {pd.DataFrame}          : The raw input DataFrame to preprocess.
        
        """
        try:
            
            if not isinstance(dataFrame, pd.DataFrame):
                preprocessor_logger.error("[DataPreprocessor] Input is not a pandas DataFrame.")
                
                raise ValueError("Input must be a pandas DataFrame.")
            
            else:
                self.df = dataFrame.copy()
                
                preprocessor_logger.info("[DataPreprocessor] DataFrame initialized successfully.")
        
        except Exception as e:
            preprocessor_logger.error(f"[DataPreprocessor] Error initializing DataFrame: {repr(e)}")
            
            raise
            

    def preprocess_data(self) -> pd.DataFrame:
        """
        Perform all preprocessing steps on the DataFrame:
        
        Steps:
            
            - Convert 'date' column to datetime format.
            - Clean HTML tags from the 'title' column.
            - Drop the 'id' column.
            - Fill missing 'subtitle' values with values from 'title'.
            - Concatenate 'title' and 'subtitle' into a new 'article' column.
            - Sort the DataFrame by 'claps' in descending order.

        Returns:
            
            pd.DataFrame    : A cleaned and preprocessed version of the input DataFrame.

        Raises:
            
            Exception       : If any error occurs during preprocessing.
        """
        try:

            self.df                = self.df.astype({'date': 'datetime64[ns]'})

            def fix_titles(title):
                
                if isinstance(title, str):
                    title = re.sub(r"<\w.*?>", "", title)
                    title = re.sub(r"</\w.*?>", "", title)
                
                return title

            self.df['title']       = self.df['title'].apply(fix_titles)
            self.df.drop(columns   = ["id"], inplace = True, errors = 'ignore')
            self.df["subtitle"].fillna(self.df["title"], inplace = True)
            self.df["article"]     = self.df["title"] + self.df["subtitle"]
            self.df                = self.df.sort_values(by = "claps", ascending = False)

            preprocessor_logger.info("[DataPreprocessor] Data preprocessing completed successfully.")

            return self.df

        except Exception as e:
            preprocessor_logger.error(f"[DataPreprocessor] Error during preprocessing: {repr(e)}")
            
            raise
