# DEPENDENCIES

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ..utils.logger import LoggerSetup

# LOGGER SETUP
vectorizer_logger = LoggerSetup(logger_name = "tfidf_vectorizer.py", log_filename_prefix = "tfidf_vectorizer").get_logger()


class Vectorizer:
    """
    A class to vectorize article texts using TF-IDF from a given DataFrame.

    This class transforms the 'article' column of a DataFrame into a sparse matrix
    of TF-IDF features which can be used for similarity calculations or modeling.

    Attributes:
    
        `df`                {pd.DataFrame}           : The input dataframe containing the 'article' column.
    
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the Vectorizer object with the provided DataFrame.

        Arguments:

            `df`              {pd.DataFrame}           : The input dataframe with a column named 'article'.

        Raises:
        
            ValueError                                 : If the input is not a pandas DataFrame.
        
        """
        
        try:
        
            if not isinstance(df, pd.DataFrame):
                vectorizer_logger.error("[Vectorizer] Input must be a pandas DataFrame.")
        
                raise ValueError("Input must be a pandas DataFrame.")

            self.df           = df
            self.vectorizer   = TfidfVectorizer()

            vectorizer_logger.info("[Vectorizer] Vectorizer initialized successfully with DataFrame.")

        except Exception as e:
            vectorizer_logger.error(f"[Vectorizer] Error during initialization: {repr(e)}")
        
            raise

    def get_vectorized_articles(self) -> csr_matrix:
        """
        Apply TF-IDF vectorization on the 'article' column of the DataFrame.

        Returns:
            
            `articles`            {csr_matrix}        : A sparse matrix representation of the TF-IDF features.

        Raises:
            
            ValueError                                : If the 'article' column does not exist in the DataFrame.
        
        """
        
        try:
        
            if 'article' not in self.df.columns:
                vectorizer_logger.error("[Vectorizer] 'article' column not found in DataFrame.")
        
                raise ValueError("'article' column not found in DataFrame.")

            articles          = self.vectorizer.fit_transform(self.df["article"])

            vectorizer_logger.info(f"[Vectorizer] Successfully vectorized {articles.shape[0]} articles.")
        
            return articles

        except Exception as e:
            vectorizer_logger.error(f"[Vectorizer] Error during vectorization: {repr(e)}")
        
            raise
