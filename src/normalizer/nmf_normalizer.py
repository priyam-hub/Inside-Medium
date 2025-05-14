# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from logger.logger import LoggerSetup

# LOGGER SETUP
nmf_logger = LoggerSetup(logger_name = "nmf_normalizer.py", log_filename_prefix = "nmf_normalizer").get_logger()


class NMFNormalizer:
    """
    A class to perform NMF decomposition and normalization on article features.

    This class takes a TF-IDF feature matrix and the original preprocessed DataFrame,
    applies Non-negative Matrix Factorization (NMF), and normalizes the result to 
    return a human-readable DataFrame indexed by article titles.

    Attributes:
    
        `articles_tfidf`       {csr_matrix}          : The TF-IDF feature matrix.
        
        `dataframe`           {pd.DataFrame}         : The original DataFrame with article metadata, including the 'title' column.
    
    """

    def __init__(self, articles_tfidf : csr_matrix, dataframe : pd.DataFrame) -> None:
        """
        Initialize the NMFNormalizer with TF-IDF features and the data.

        Arguments:
            
            `articles_tfidf`          {csr_matrix}         : The sparse TF-IDF feature matrix.
            
            `dataframe`              {pd.DataFrame}        : The preprocessed DataFrame including the 'title' column.

        Raises:
            
            ValueError: If inputs are not of expected types or 'title' column is missing.
        
        """
        
        try:
        
            if not isinstance(articles_tfidf, csr_matrix):
                nmf_logger.error("[NMFNormalizer] Input TF-IDF features must be a csr_matrix.")
                
                raise ValueError("TF-IDF features must be a csr_matrix.")

            if not isinstance(dataframe, pd.DataFrame):
                nmf_logger.error("[NMFNormalizer] Input data must be a pandas DataFrame.")
                
                raise ValueError("Data must be a pandas DataFrame.")

            if 'title' not in dataframe.columns:
                nmf_logger.error("[NMFNormalizer] 'title' column not found in DataFrame.")
                
                raise ValueError("'title' column must be present in DataFrame.")

            self.articles_tfidf = articles_tfidf
            self.data           = dataframe

            nmf_logger.info("[NMFNormalizer] Initialization successful.")

        except Exception as e:
            nmf_logger.error(f"[NMFNormalizer] Error during initialization: {repr(e)}")
            
            raise

    def normalize_nmf_features(self, n_components : int = 10) -> pd.DataFrame:
        """
        Apply NMF and normalize the resulting features.

        Arguments:

            n_components            {int}            : Number of latent topics/components to use in NMF. Default is 10.

        Returns:
        
            df_normalized       {pd.DataFrame}       : A DataFrame of normalized NMF features indexed by article titles.
        
        Raises:
        
            Exception: If NMF or normalization fails.
        
        """
        
        try:
        
            nmf_logger.info(f"[NMFNormalizer] Applying NMF with {n_components} components.")
            
            model           = NMF(n_components = n_components, random_state = 0)
            nmf_features    = model.fit_transform(self.articles_tfidf)

            nmf_logger.info("[NMFNormalizer] NMF transformation successful. Normalizing features.")
            normalized      = normalize(nmf_features)

            df_normalized   = pd.DataFrame(data = normalized, index = self.data["title"])

            nmf_logger.info("[NMFNormalizer] Normalized features DataFrame created successfully.")
            
            return df_normalized

        except Exception as e:
            nmf_logger.error(f"[NMFNormalizer] Error during NMF normalization: {repr(e)}")
            
            raise
