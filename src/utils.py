import os
import sys
import numpy as np
import pandas as pd
import dill
from src.logger import logging
from src.exception import CustomException


def exception_handler(func):
    def handle(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise
    return handle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.error(CustomException(e, sys))
        raise