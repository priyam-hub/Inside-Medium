# DEPENDENCIES

import os
import logging
from pathlib import Path

logging.basicConfig(level = logging.INFO, format = '[%(asctime)s]: %(message)s:')

list_of_paths = [
    ".dockerignore",
    ".env",
    ".gitignore",
    "Dockerfile",
    "env_setup.sh",
    "folder_structure.py",
    "LICENSE",
    "main.py",
    "README.md",
    "requirements.txt",
    "setup.py",
    "config/__init__.py",
    "config/config.py",
    "data/",
    "notebooks/Recommendation_System.ipynb",
    "src/data_preprocessor/__init__.py",
    "src/data_preprocessor/data_preprocessor.py",
    "src/exploratory_data_analysis/__init__.py",
    "src/exploratory_data_analysis/exploratory_data_analyzer.py",
    "src/utils/__init__.py",
    "src/utils/load_data.py",
    "src/utils/save_plot.py",
    "src/utils/logger.py",
    "web/__init__.py",
    "web/static/style.css",
    "web/static/script.js",
    "web/templates/index.html",
    "web/app.py"
]


for path_str in list_of_paths:
    path = Path(path_str)

    if path_str.endswith("/"):  # Directory
        
        if not path.exists():
            os.makedirs(path, exist_ok=True)
            logging.info(f"Created directory: {path}")
        
        else:
            logging.info(f"Directory already exists: {path}")
    
    else:  # File
    
        if not path.exists():
    
            if not path.parent.exists():
                os.makedirs(path.parent, exist_ok=True)
                logging.info(f"Created parent directory: {path.parent} for the file: {path.name}")
    
            with open(path, "w") as f:
                pass
            logging.info(f"Created empty file: {path}")
    
        else:
            logging.info(f"File already exists: {path}")