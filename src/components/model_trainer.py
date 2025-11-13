import os
import sys
from dataclasses import dataclass
from pathlib import Path

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        project_root = Path(__file__).resolve().parents[2]
        artifacts_dir = project_root / 'artifacts'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_trainer_config.trained_model_file_path = str(artifacts_dir / 'model.pkl')


    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
                "Adaboost": AdaBoostRegressor(),
                "KNeighbors": KNeighborsRegressor() 
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

        
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info("Best model found")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

        except Exception as e:
            raise CustomException(e,sys)