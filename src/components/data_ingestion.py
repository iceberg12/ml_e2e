import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import exception_handler


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
        # os.makedirs(self.ingestion_config.train_data_path, exist_ok=True)
        # os.makedirs(self.ingestion_config.test_data_path, exist_ok=True)

    @exception_handler
    def init_data_ingestion(self):
        logging.info('Enter data ingestion')

        df = pd.read_csv('notebook/data/stud.csv')
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=12)

        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        logging.info('Ingestion is completed.')

        return (self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.init_data_ingestion()