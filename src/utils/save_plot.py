# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from librosa import ex
from logger.logger import LoggerSetup
import matplotlib.pyplot as plt

# LOGGER SETUP
save_plot_logger = LoggerSetup(logger_name = "save_plot.py", log_filename_prefix = "save_plot").get_logger()

class PlotSaver:
        
    """
    A utility class for saving matplotlib plots to disk with logging support.

    This class is designed to be used in data analysis or machine learning pipelines
    where visualizations are generated and need to be persistently stored.

    Arguments:
    
        output_dir              {str}           : The directory where plots should be saved. If None, plots are not saved.
        
    """

    def __init__(self, output_dir : str = None) -> None:

        """
        Initialize the PlotSaver object.

        Arguments:
        
            output_dir           {str, optional}    : The path to the directory where plots should be saved.

        Returns:

            None
        
        """
        
        self.output_dir = output_dir

    def save_plot(self, plot : plt, plot_name : str) -> None:
        """
        Save the plot to disk if an output directory is specified.

        Arguments:
        
            `plt`             {matplotlib.pyplot}           : The plot object to save.
            
            `plot_name`              {str}                  : Name for the output file.

        Returns:

            None
        
        """

        try:
        
            if self.output_dir:
                
                plot.savefig(fname = f"{self.output_dir}/{plot_name}.png", bbox_inches = 'tight', dpi = 300)
                save_plot_logger.info(f"Saved plot: {plot_name}")

        except Exception as e:
            save_plot_logger.error(f"Error saving plot: {repr(e)}")
            
            raise e