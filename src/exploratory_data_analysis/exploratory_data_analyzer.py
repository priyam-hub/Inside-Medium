# DEPENDENCIES

import re
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
from librosa import ex
from pathlib import Path
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords

from ..utils.logger import LoggerSetup
from ..utils.save_plot import PlotSaver

# LOGGER SETUP
eda_logger = LoggerSetup(logger_name = "exploratory_data_analyzer.py", log_filename_prefix = "exploratory_data_analyzer").get_logger()


class MediumEDA:
    
    """
    A class for comprehensive Exploratory Data Analysis (EDA) of Medium Article Recommendation Dataset
    
    Attributes:

        `df`                     {pd.DataFrame}         : DataFrame containing text and emotion data.

        `output_dir`                  {str}             : Optional directory to save generated plots.
    
    """

    def __init__(self, dataframe : pd.DataFrame, output_dir : str = None) -> None:
        """
        Initialize the MediumEDA class.
        
        Arguments:

            `df`                   {pd.DataFrame}         : Input DataFrame with text and emotion data.
            
            `output_dir`                {str}             : Optional path to save plots.
        
        Raises:

            ValueError                                    : If required columns are not found in the DataFrame.
        
        Returns:

            None
        
        """
        
        try:

            if not isinstance(dataframe, pd.DataFrame):
                eda_logger.error("Input data is not a pandas DataFrame")
                
                raise 

            else:
                eda_logger.info(f"DataFrame Loaded with shape: {dataframe.shape}")

            self.df               = dataframe
            self.output_dir       = output_dir
            self.plt_saver        = PlotSaver(output_dir = output_dir)

            if output_dir:
                Path(output_dir).mkdir(parents = True, exist_ok = True)

                eda_logger.info(f"Output directory created: {output_dir}")

            eda_logger.info(f"MediumEDA Class initialized")

        except Exception as e:
            eda_logger.error(f"Error initializing EmotionEDA: {repr(e)}")
            
            raise

    def plot_reading_time_ecdf(self, reading_time_column : str = 'reading_time') -> None:
        """
        Plot the ECDF of reading time compared with a theoretical normal distribution.

        Arguments:
            
            `reading_time_column`            {str}        : Column name containing reading time data.

        Returns:
            
            None
        
        """
        
        try:
            if reading_time_column not in self.df.columns:
                eda_logger.error(f"Column '{reading_time_column}' not found in DataFrame.")
                
                raise ValueError(f"Column '{reading_time_column}' not found in DataFrame.")

            data_col         = self.df[reading_time_column].dropna()
            
            # Empirical CDF function
            def ecdf(data):
                x            = np.sort(data)
                y            = np.arange(1, len(x) + 1) / len(x)
                
                return x, y

            sample           = np.random.normal(loc = data_col.mean(), scale = data_col.std(), size = 10000)

            x, y             = ecdf(data_col)
            x_theor, y_theor = ecdf(sample)

            plt.figure(figsize = (10, 6))

            plt.plot(x, y, 
                     marker     = "o", 
                     linestyle  = "none", 
                     label      = "Data Distribution", 
                     color      = "#1f77b4", 
                     alpha      = 0.8
                     )

            plt.plot(x_theor, y_theor, 
                     color      = "#d62728", 
                     linestyle  = "--", 
                     linewidth  = 2, 
                     label      = "Theoretical Normal Distribution"
                     )

            plt.title("Empirical vs Theoretical ECDF of Reading Time", fontsize = 16, fontweight = 'bold', color = "#333333")
            plt.xlabel("Reading Time", fontsize = 12, labelpad = 10)
            plt.ylabel("ECDF", fontsize = 12, labelpad = 10)

            plt.grid(True, 
                     linestyle  = '--', 
                     linewidth  = 0.5, 
                     alpha      = 0.7
                     )

            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.legend(fontsize     = 10, 
                       loc          = 'lower right', 
                       frameon      = True, 
                       framealpha   = 0.9, 
                       facecolor    = 'white', 
                       edgecolor    = 'gray'
                       )

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            self.plt_saver.save_plot(plot = plt, plot_name = "ecdf_reading_time")

            eda_logger.info("Beautifully styled ECDF plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating ECDF plot: {repr(e)}")
            
            raise

    def plot_reading_time_histogram(self, reading_time_column : str = 'reading_time', bins : int = 100) -> None:
        """
        Plot a histogram of the reading time.

        Arguments:
            
            `reading_time_column`           {str}          : Column name containing reading time data.
            
            `bins`                          {int}          : Number of histogram bins (default is 100).

        Returns:
            
            None
        
        """
        try:
            
            if reading_time_column not in self.df.columns:
                eda_logger.error(f"Column '{reading_time_column}' not found in DataFrame.")
                
                raise ValueError(f"Column '{reading_time_column}' not found in DataFrame.")

            data_col = self.df[reading_time_column].dropna()

            plt.figure(figsize = (10, 6))

            plt.hist(data_col,
                     bins       = bins,
                     color      = "#4C72B0",
                     edgecolor  = 'white',
                     alpha      = 0.85
                     )

            plt.title("Distribution of Reading Time", fontsize = 16, fontweight = 'bold', color = '#333333')
            plt.xlabel("Reading Time (in minutes)", fontsize = 12, labelpad = 10)
            plt.ylabel("Frequency", fontsize = 12, labelpad = 10)

            plt.grid(True, linestyle = '--', alpha = 0.5)

            plt.xticks(fontsize = 10)
            plt.yticks(fontsize = 10)

            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

            plt.tight_layout()

            self.plt_saver.save_plot(plot = plt, plot_name = "reading_time_histogram")

            eda_logger.info("Histogram plot for reading_time created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating histogram plot: {repr(e)}")
            
            raise

    def plot_continuous_vs_discrete_reading_time(self, reading_time_column : str = 'reading_time') -> None:
        """
        Plot a side-by-side comparison of continuous (Exponential) and discrete-like (Empirical) 
        distributions for reading time.

        Arguments:
            
            `reading_time_column`           {str}          : Column name containing reading time data.

        Returns:
            
            None
        """
        
        try:
            
            if reading_time_column not in self.df.columns:
                eda_logger.error(f"Column '{reading_time_column}' not found in DataFrame.")
                
                raise ValueError(f"Column '{reading_time_column}' not found in DataFrame.")

            data_col = self.df[reading_time_column].dropna()

            
            plt.figure(figsize = (12, 6))

            plt.subplot(1, 2, 1)
            exp_sample = np.random.exponential(scale = data_col.mean(), size = 6000)
            plt.hist(exp_sample, bins = 100, color = 'skyblue', edgecolor = 'black')
            plt.title("Exponential Distribution (Continuous)", fontsize = 12)
            plt.xlabel("Reading Time", fontsize = 10)
            plt.ylabel("Frequency", fontsize = 10)

            plt.subplot(1, 2, 2)
            plt.hist(data_col, bins = 100, color = 'salmon', edgecolor = 'black')
            plt.title("Empirical Distribution (Observed Data)", fontsize = 12)
            plt.xlabel("Reading Time", fontsize = 10)
            plt.ylabel("Frequency", fontsize = 10)

            plt.suptitle("Continuous vs Discrete-like Reading Time Distributions", fontsize = 14, fontweight = 'bold')

            plt.tight_layout(rect = [0, 0.03, 1, 0.95])  
            self.plt_saver.save_plot(plot = plt, plot_name = "continuous_vs_discrete_reading_time")

            eda_logger.info("Continuous vs Discrete-like distribution plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating continuous vs discrete plot: {repr(e)}")
            
            raise

    
    def plot_poisson_ecdf_comparison(self, reading_time_column: str = 'reading_time', sample_size: int = 6000) -> None:
        """
        Plot ECDF of actual reading_time data and compare it with theoretical Poisson distribution.

        Arguments:
            
            `reading_time_column`           {str}          : Column name containing reading time data.
            
            `sample_size`                   {int}          : Size of Poisson sample to generate (default is 6000).

        Returns:
            
            None
        
        """
        
        try:
        
            if reading_time_column not in self.df.columns:
                eda_logger.error(f"Column '{reading_time_column}' not found in DataFrame.")
                
                raise ValueError(f"Column '{reading_time_column}' not found in DataFrame.")

            data_col = self.df[reading_time_column].dropna()

            def ecdf(data):
                x    = np.sort(data)
                y    = np.arange(1, len(data) + 1) / len(data)
                
                return x, y

            x, y     = ecdf(data_col)

            poisson  = np.random.poisson(data_col.mean(), size = sample_size)
            x_theor_po, y_theor_po = ecdf(poisson)

            plt.figure(figsize = (10, 6))

            plt.plot(x, y, 
                     marker     = ".", 
                     linestyle  = "none", 
                     label      = "Data ECDF", 
                     color      = "#1f77b4")
            
            plt.plot(x_theor_po, y_theor_po, 
                     color      = "crimson", 
                     linewidth  = 2, 
                     label      = "Theoretical Poisson ECDF"
                     )

            plt.title("ECDF Comparison with Poisson Distribution", 
                      fontsize    = 16, 
                      fontweight  = 'bold', 
                      color       = '#333333'
                      )
            
            plt.xlabel("Reading Time", 
                       fontsize  = 12, 
                       labelpad  = 10
                       )
            
            plt.ylabel("ECDF", 
                       fontsize  = 12, 
                       labelpad  = 10
                       )
            
            plt.legend()
            plt.grid(True, 
                     linestyle  = '--', 
                     linewidth  = 0.6, 
                     alpha      = 0.7
                     )

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            self.plt_saver.save_plot(plot = plt, plot_name = "poisson_ecdf_comparison")

            eda_logger.info("ECDF comparison with Poisson distribution plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating Poisson ECDF comparison plot: {repr(e)}")
            
            raise

    def plot_publication_count(self, publication_column: str = 'publication') -> None:
        """
        Create a horizontal count plot for the publication column.

        Arguments:
        
            publication_column {str} : Column name containing publication names (default is 'publication').

        Returns:
        
            None
        
        """
        
        try:
        
            if publication_column not in self.df.columns:
                eda_logger.error(f"Column '{publication_column}' not found in DataFrame.")
        
                raise ValueError(f"Column '{publication_column}' not found in DataFrame.")

            data_col = self.df[publication_column].dropna()

            plt.figure(figsize = (10, 6))
            sns.set_style("whitegrid")
            sns.set_palette("viridis")

            sns.countplot(y      = publication_column,
                          data   = self.df,
                          order  = data_col.value_counts().index)

            plt.title("Article Count by Publication", 
                      fontsize    = 16, 
                      fontweight  = 'bold', 
                      color       ='#333333'
                      )
            
            plt.xlabel("Count", fontsize=12)
            plt.ylabel("Publication", fontsize=12)
            plt.grid(axis       = 'x', 
                     linestyle  = '--', 
                     alpha      = 0.5
                     )

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            self.plt_saver.save_plot(plot = plt, plot_name = "publication_count_plot")

            eda_logger.info("Publication count plot created successfully")

        except Exception as e:
            eda_logger.error(f"Error creating publication count plot: {repr(e)}")
            
            raise


    def run_all_eda(self):
        """
        Function to run all EDA functions in sequence.

        """

        try:

            self.plot_reading_time_ecdf()
            self.plot_reading_time_histogram()
            self.plot_continuous_vs_discrete_reading_time()
            self.plot_poisson_ecdf_comparison()
            self.plot_publication_count()

            eda_logger.info("All EDA plots generated successfully")

        except Exception as e:
            eda_logger.error(f"Error running all EDA functions: {repr(e)}")
            
            raise
