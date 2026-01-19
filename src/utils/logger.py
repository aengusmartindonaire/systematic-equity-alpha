import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name="systematic_alpha", log_dir="logs"):
    """
    Sets up a logger that writes to console and a file.
    """
    # Create logs directory if it doesn't exist (relative to where script runs)
    # Ideally, use the config paths, but this is a safe fallback
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Log filename with date
    log_filename = f"{log_dir}/run_{datetime.now().strftime('%Y%m%d')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs in Jupyter
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger