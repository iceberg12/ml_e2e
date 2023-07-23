import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, exception_handler
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    @exception_handler
    def init_model_trainer(self, train_data, test_data):
        logging.info('Model training started.')
        X_train, y_train, X_test, y_test = (
            train_data[:, :-1], train_data[:, -1],
            test_data[:, :-1], test_data[:, -1]
        )
        models = {
            'Random forest': RandomForestRegressor(),
            'Decision tree': DecisionTreeRegressor(),
            'Gradient boosting': GradientBoostingRegressor(),
            'Linear regression': LinearRegression(),
            'K-NN classifier': KNeighborsRegressor(),
            'XGB classifier': XGBRegressor(),
            'Catboost classifer': CatBoostRegressor(verbose=False, allow_writing_files=False),
            'Adaboost classifer': AdaBoostRegressor()
        }
        model_report = self.evaluate_models(
            X_train, y_train, X_test, y_test, models
        )
        test_report = {name: errors['test_model_score']
                       for name, errors in model_report.items()}
        best_model_name = max(test_report, key=test_report.get)
        best_model = models[best_model_name]
        if test_report[best_model_name] < 0.6:
            logging.error(CustomException('No best model found', sys))

        logging.info(f'Best model found: {best_model_name}'
                     f'with R2 score {test_report[best_model_name]}')
        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        return best_model

    @exception_handler
    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        report = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                'train_model_score': train_model_score,
                'test_model_score': test_model_score,
            }
        return report

if __name__ == '__main__':
    di = DataIngestion()
    train_path, test_path = di.init_data_ingestion()

    dt = DataTransformation()
    train_data, test_data, _ = dt.init_transformation(train_path, test_path)

    mt = ModelTrainer()
    mt.init_model_trainer(train_data, test_data)
