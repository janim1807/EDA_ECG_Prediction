import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
import joblib


def load_split_data(prefix):
    """
    Load training, validation, and test data for the given prefix (regression or categorical).

    Args:
    prefix (str): The prefix for the file paths.

    Returns:
    tuple: DataFrames and Series for training, validation, and test sets.
    """
    X_train = pd.read_csv(f'../data/modeling/train/{prefix}_X_train.csv')
    X_val = pd.read_csv(f'../data/modeling/dev/{prefix}_X_val.csv')
    X_test = pd.read_csv(f'../data/modeling/test/{prefix}_X_test.csv')
    y_train = pd.read_csv(f'../data/modeling/train/{prefix}_y_train.csv').squeeze()
    y_val = pd.read_csv(f'../data/modeling/dev/{prefix}_y_val.csv').squeeze()
    y_test = pd.read_csv(f'../data/modeling/test/{prefix}_y_test.csv').squeeze()

    return X_train, X_val, X_test, y_train, y_val, y_test


def define_models_and_params():
    """
    Define regression models and their hyperparameter grids.

    Returns:
    tuple: Dictionary of models and dictionary of hyperparameter grids.
    """
    param_grids = {
        'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0, 200, 1000, 2000]},
        'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
        'ElasticNet': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]},
        'GradientBoosting': {'n_estimators': [50, 100, 200, 500], 'learning_rate': [0.001, 0.01, 0.1, 0.2]},
        'SVR': {'C': [1, 10, 100, 200], 'epsilon': [0.1, 0.2, 0.5, 1, 1.5, 2]},
        'XGBRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9, 12]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,), (100, 100)], 'alpha': [0.0001, 0.001, 0.01]}
    }

    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGBRegressor': XGBRegressor(),
        'MLPRegressor': MLPRegressor(max_iter=10000)
    }

    return models, param_grids


def define_classification_models_and_params():
    """
    Define classification models and their hyperparameter grids.

    Returns:
    tuple: Dictionary of models and dictionary of hyperparameter grids.
    """
    param_grids = {
        'LogisticRegression': {'C': [0.01, 0.1, 1, 10, 100]},
        'SVC': {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'RandomForest': {'n_estimators': [50, 100, 200, 400], 'max_depth': [None, 10, 20, 30]},
        'GradientBoosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
        'KNeighbors': {'n_neighbors': [3, 5, 7, 9]}
    }

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVC': SVC(probability=True),
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'KNeighbors': KNeighborsClassifier()
    }

    return models, param_grids


def perform_grid_search(models, param_grids, X_train, y_train):
    """
    Perform grid search for each model to find the best hyperparameters.

    Args:
    models (dict): Dictionary of models.
    param_grids (dict): Dictionary of hyperparameter grids.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.

    Returns:
    dict: Dictionary of best models after grid search.
    """
    best_models = {}
    for name, model in models.items():
        print(f"Performing grid search for {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    return best_models


def perform_classification_grid_search(models, param_grids, X_train, y_train):
    """
    Perform grid search for each classification model to find the best hyperparameters.

    Args:
    models (dict): Dictionary of models.
    param_grids (dict): Dictionary of hyperparameter grids.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target variable.

    Returns:
    dict: Dictionary of best models after grid search.
    """
    best_models = {}
    for name, model in models.items():
        print(f"Performing grid search for {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    return best_models


def evaluate_models(best_models, X_val, y_val):
    """
    Evaluate the best models on the validation set.

    Args:
    best_models (dict): Dictionary of best models after grid search.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target variable.

    Returns:
    dict: Dictionary of evaluation results for each model.
    """
    results = {}
    for name, model in best_models.items():
        y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(y_val, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        r2_val = r2_score(y_val, y_val_pred)
        results[name] = {
            'MSE': mse_val,
            'RMSE': rmse_val,
            'MAE': mae_val,
            'R2': r2_val
        }
        print(
            f"{name} - Validation Metrics: MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R2: {r2_val:.4f}")
    return results


def evaluate_classification_models(best_models, X_val, y_val):
    """
    Evaluate the best classification models on the validation set.

    Args:
    best_models (dict): Dictionary of best models after grid search.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target variable.

    Returns:
    dict: Dictionary of evaluation results for each model.
    """
    results = {}
    for name, model in best_models.items():
        y_val_pred = model.predict(X_val)
        accuracy_val = accuracy_score(y_val, y_val_pred)
        precision_val = precision_score(y_val, y_val_pred, average='macro')
        recall_val = recall_score(y_val, y_val_pred, average='macro')
        f1_val = f1_score(y_val, y_val_pred, average='macro')
        results[name] = {
            'Accuracy': accuracy_val,
            'Precision': precision_val,
            'Recall': recall_val,
            'F1': f1_val,
        }
        print(
            f"{name} - Validation Metrics: Accuracy: {accuracy_val:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}, F1: {f1_val:.4f}")
    return results


def save_best_model(best_model, best_model_name, folder_path):
    """
    Save the best model to the specified folder.

    Args:
    best_model: The best model object.
    best_model_name (str): The name of the best model.
    folder_path (str): The path to the folder to save the model.
    """
    file_path = os.path.join(folder_path, f'{best_model_name}_model.pkl')
    joblib.dump(best_model, file_path)
    print(f"Best model saved to {file_path}")


def main():
    """
    Main function to execute the model training, evaluation, and saving.
    """
    # Load the regression data
    X_train_reg, X_val_reg, X_test_reg, y_train_reg, y_val_reg, y_test_reg = load_split_data('regression_augmented')

    # Prepare the data for regression model training
    X_train_reg = X_train_reg.drop(columns=['Participant'])
    X_val_reg = X_val_reg.drop(columns=['Participant'])

    # Define regression models and parameter grids
    models_reg, param_grids_reg = define_models_and_params()

    # Perform grid search for regression models
    best_models_reg = perform_grid_search(models_reg, param_grids_reg, X_train_reg, y_train_reg)

    # Evaluate regression models
    results_reg = evaluate_models(best_models_reg, X_val_reg, y_val_reg)

    # Select the best regression model based on validation performance
    best_model_name_reg = min(results_reg, key=lambda k: results_reg[k]['MSE'])
    best_model_reg = best_models_reg[best_model_name_reg]
    print(f"Best Regression Model: {best_model_name_reg}")

    # Save the best regression model
    save_best_model(best_model_reg, best_model_name_reg, '../models')

    # Load the classification data
    X_train_cat, X_val_cat, X_test_cat, y_train_cat, y_val_cat, y_test_cat = load_split_data('categorical_augmented')

    # Prepare the data for classification model training
    X_train_cat = X_train_cat.drop(columns=['Participant'])
    X_val_cat = X_val_cat.drop(columns=['Participant'])

    # Define classification models and parameter grids
    models_cat, param_grids_cat = define_classification_models_and_params()

    # Perform grid search for classification models
    best_models_cat = perform_classification_grid_search(models_cat, param_grids_cat, X_train_cat, y_train_cat)

    # Evaluate classification models
    results_cat = evaluate_classification_models(best_models_cat, X_val_cat, y_val_cat)

    # Select the best classification model based on validation performance
    best_model_name_cat = max(results_cat, key=lambda k: results_cat[k]['Accuracy'])
    best_model_cat = best_models_cat[best_model_name_cat]
    print(f"Best Classification Model: {best_model_name_cat}")

    # Save the best classification model
    save_best_model(best_model_cat, best_model_name_cat, '../models')


# Run the main function
if __name__ == "__main__":
    main()
