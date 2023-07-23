import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.transformation_config = DataTransformationConfig()

    def get_transformer(self):
        try:
            num_cols = ['writing_score', 'reading_score']
            cat_cols = ['gender', 'race_ethnicity',
                        'parental_level_of_education', 'lunch',
                        'test_preparation_course']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder()),
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols),
            ])
            return preprocessor
        except Exception as e:
            logging.error(CustomException(e, sys))
            raise

    def init_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test data are loaded.')

            preprocessor = self.get_transformer()

            target_col = 'math_score'
            features_train=train_df.drop(columns=[target_col],axis=1)
            target_train=train_df[target_col]

            features_test=test_df.drop(columns=[target_col],axis=1)
            target_test=test_df[target_col]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_np=preprocessor.fit_transform(features_train)
            test_np=preprocessor.transform(features_test)

            train_np = np.c_[train_np, np.array(target_train)]
            test_np = np.c_[test_np, np.array(target_test)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.transformation_config.preprocessor_path,
                obj=preprocessor
            )

            return (train_np,
                    test_np,
                    self.transformation_config.preprocessor_path)

        except Exception as e:
            logging.error(CustomException(e, sys))
            raise

if __name__ == '__main__':
    di = DataIngestion()
    train_path, test_path = di.init_data_ingestion()

    dt = DataTransformation()
    train_np, test_np, preprocessor_path = dt.init_transformation(train_path, test_path)

