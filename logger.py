import logging, os
from datetime import datetime
import json


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
    # formatter = logging.Formatter('%(levelname)s %(custom_tag)s: %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def log_info(logger, message):
    logger.info(message)


def log_entry(logger, instance, name, origina_outcome, predicted_outcome):
    features = {k: v.item() if hasattr(v, "item") else v for k, v in instance.to_dict().items()}
    log_entry = {
        "type": "entry",
        "entry_name": str(name),
        "instance": {
            "features": features,
            "original_outcome": origina_outcome,
            "predicted_outcome": predicted_outcome
        }
    }
    logger.info(json.dumps(log_entry))


def log_rule(logger, rule, name, measures):
    rule_log = {
        "type": "rule",
        "entry_name": str(name),
        "explainer": rule.explainer,
        "at_limit": rule.at_limit,
        "rule": {
            "premises": rule.premises,
            "consequence": rule.consequence,
        },
        "metrics": measures,
    }
    logger.info(json.dumps(rule_log))


def log_with_custom_tag(logger, message, tag = ""):
    logger.info(message, extra={'custom_tag': tag})