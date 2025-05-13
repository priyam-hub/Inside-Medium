# DEPENDENCIES

import pandas as pd
from pandas import DataFrame
from difflib import get_close_matches

from ..utils.logger import LoggerSetup

# LOGGER SETUP
similarity_logger = LoggerSetup(logger_name = "similarity_finder.py", log_filename_prefix = "similarity_finder").get_logger()


class SimilarityFinder:
    """
    A class to compute similarity between articles based on normalized NMF features.

    This class takes a normalized article feature DataFrame and provides a function to find
    the most similar articles to a given query title.

    Attributes:
        
        `df_normalized`             {pd.DataFrame}            : The normalized NMF features DataFrame indexed by article titles.
        
        `preprocessed_data`         {pd.DataFrame}            : The original metadata DataFrame (must contain 'title' and 'claps').
    
    """

    def __init__(self, df_normalized : DataFrame, preprocessed_data : DataFrame) -> None:
        """
        Initialize the SimilarityFinder with normalized data and metadata.

        Arguments:

            `df_normalized`           {pd.DataFrame}            : The normalized NMF features with article titles as index.
        
            `preprocessed_data`       {pd.DataFrame}            : The original dataset containing 'title' and 'claps'.

        Raises:
            
            ValueError: If required inputs are invalid or missing required columns.
        
        """
        
        try:
        
            if not isinstance(df_normalized, pd.DataFrame):
                similarity_logger.error("[SimilarityFinder] Normalized data must be a pandas DataFrame.")
        
                raise ValueError("Normalized data must be a DataFrame.")

            if not isinstance(preprocessed_data, pd.DataFrame):
                similarity_logger.error("[SimilarityFinder] Metadata must be a pandas DataFrame.")
        
                raise ValueError("Metadata must be a DataFrame.")

            if "title" not in preprocessed_data.columns or "claps" not in preprocessed_data.columns:
                similarity_logger.error("[SimilarityFinder] 'title' or 'claps' columns missing from metadata.")
        
                raise ValueError("Metadata must include 'title' and 'claps' columns.")

            self.df_normalized   = df_normalized
            self.data            = preprocessed_data

            similarity_logger.info("[SimilarityFinder] Initialization successful.")

        except Exception as e:
            similarity_logger.error(f"[SimilarityFinder] Error during initialization: {repr(e)}")
            
            raise

    def get_similar_articles(self, query_title : str, top_n : int = 10) -> pd.DataFrame:
        """
        Find top-N similar articles based on the query article title.

        Arguments:

            `query_title`          {str}            : The title of the article to find similarities against.
            
            `top_n`                {int}            : Number of top similar articles to return. Default is 10.

        Returns:

            pd.DataFrame                            : Top-N most similar articles, merged with claps and sorted by similarity 
                                                      (descending).

        Raises:
            
            ValueError: If the query title is not found in the normalized data.
        
        """
        
        try:
        
            if query_title not in self.df_normalized.index:

                matches = get_close_matches(query_title, self.df_normalized.index, n = 1, cutoff = 0.6)

                if matches:
                    match_title = matches[0]
                    similarity_logger.warning(f"[SimilarityFinder] Query title '{query_title}' not found. Using closest match: '{match_title}'")
        
                    query_title = match_title
                
                else:
                    similarity_logger.error(f"[SimilarityFinder] Query title '{query_title}' not found in index and no close matches.")
                    
                    raise ValueError(f"Query title '{query_title}' not found in normalized DataFrame.")

            similarity_logger.info(f"[SimilarityFinder] Computing similarity for: {query_title}")

            current             = self.df_normalized.loc[query_title]
            similarities        = self.df_normalized.dot(current)
            sims_df             = pd.DataFrame(similarities.nlargest(top_n), columns = ["similarity"])
            sims_df             = sims_df.merge(self.data[["title", "claps"]], how = "inner", on = "title")
            
            sims_df.set_index("title", drop = True, inplace = True)

            similarity_logger.info(f"[SimilarityFinder] Similar articles found successfully for: {query_title}")
            
            return sims_df.sort_values(by = "similarity", ascending = False)

        except Exception as e:
            similarity_logger.error(f"[SimilarityFinder] Error in get_similar_articles: {repr(e)}")
            
            raise
