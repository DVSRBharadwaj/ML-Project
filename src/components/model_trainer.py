import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training & test input data')
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Classifier': KNeighborsRegressor(),
                'XGBClassifier': XGBRegressor(),
                'CatBoosting Classifier': CatBoostRegressor(random_state=42, 
                                                            #verbose=False, 
                                                            logging_level='Silent',
                                                            thread_count=-1),
                'AdaBoost Classifier': AdaBoostRegressor()
            }
            param_grids = {
                "Random Forest": {
                    "n_estimators": [300, 400, 500, 600],
                    "max_depth": [6, 7, 8, 9, 10, None],
                    #"min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4],
                    #"max_features": ["sqrt", "log2", 0.7, 0.8],
                },

                "Gradient Boosting": {
                    "n_estimators": [300, 400, 500, 600],
                    "learning_rate": [0.01, 0.03, 0.05, 0.08],
                    #"max_depth": [3, 4, 5, 6],
                    #"min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4],
                    #"subsample": [0.7, 0.8, 0.9, 1.0],
                },

                "XGBClassifier": {
                    "n_estimators": [400, 500, 600, 700, 800],
                    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                    #"max_depth": [4, 5, 6, 7],
                    #"min_child_weight": [1, 3, 5],
                    #"subsample": [0.7, 0.8, 0.9, 1.0],
                    #"colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                    #"gamma": [0, 0.1, 0.2],
                    #"reg_alpha": [0, 0.1, 0.5],
                    #"reg_lambda": [0.5, 1.0, 1.5],
                },

                "CatBoosting Classifier": {
                    "iterations": [500, 700, 800, 1000],
                    "learning_rate": [0.01, 0.03, 0.05, 0.07],
                    #"depth": [4, 5, 6, 7, 8],
                    #"l2_leaf_reg": [1, 3, 5, 7],
                    #"border_count": [128, 254],
                    #"bagging_temperature": [0.0, 0.2, 0.5, 1.0],
                    #"random_strength": [0.5, 1, 2],
                },

                "K-Neighbors Classifier": {
                    "n_neighbors": [5, 7, 9, 11, 13, 15],
                    "weights": ["uniform", "distance"],
                    #"metric": ["euclidean", "manhattan", "minkowski"],
                    #"p": [1, 2]  # only used if metric='minkowski'
                },

                "AdaBoost Classifier": {
                    "n_estimators": [200, 300, 400, 500],
                    "learning_rate": [0.05, 0.1, 0.3, 0.5, 1.0],
                    #"loss": ["linear", "square", "exponential"],
                    #"base_estimator": [
                        #DecisionTreeRegressor(max_depth=3),
                       # DecisionTreeRegressor(max_depth=4),
                        #DecisionTreeRegressor(max_depth=5)
                    #]
                },

                "Decision Tree": {
                    "max_depth": [4, 5, 6, 7, 8, 10, None],
                    "min_samples_split": [2, 5, 10, 20],
                    #"min_samples_leaf": [1, 2, 4, 8],
                    #"max_features": ["sqrt", "log2", None]
                },

                # LinearRegression has no hyperparameters to tune
                "Linear Regression": {}
            }

            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train,
                                                x_test=x_test, y_test=y_test, models=models, param=param_grids)
            # To get the best model score 
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found!')
            logging.info('Best model found on both training & test dataset.')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            raise CustomException(e, sys)