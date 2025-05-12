import logging, os
from datetime import datetime

def setup_logger(log_file, name):
    logger = logging.getLogger(name)  # create a named logger for each dataset
    logger.setLevel(logging.INFO)

    # clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    timestamp = datetime.now().strftime('%M-%H_%d-%m-%Y')
    base, ext = os.path.splitext(log_file)
    timestamped_log_file = f"{base}_{timestamp}{ext}"

    # create file handler
    file_handler = logging.FileHandler(timestamped_log_file, mode='w')  # 'w' to overwrite, 'a' to append
    # formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(custom_tag)s: %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(levelname)s %(custom_tag)s: %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def log_with_custom_tag(logger, message, tag = ""):
    logger.info(message, extra={'custom_tag': tag})