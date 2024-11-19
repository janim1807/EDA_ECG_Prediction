import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


from src.train import (load_split_data, define_models_and_params,
                       define_classification_models_and_params,
                       perform_grid_search, perform_classification_grid_search,
                       evaluate_models, evaluate_classification_models, save_best_model)


class TestModelPipeline(unittest.TestCase):

    @patch("src.train.pd.read_csv")
    def test_load_split_data(self, mock_read_csv):
        # Setup mock return values
        mock_read_csv.side_effect = [
            pd.DataFrame(np.random.rand(100, 5)),  # X_train
            pd.DataFrame(np.random.rand(20, 5)),  # X_val
            pd.DataFrame(np.random.rand(20, 5)),  # X_test
            pd.Series(np.random.rand(100)),  # y_train
            pd.Series(np.random.rand(20)),  # y_val
            pd.Series(np.random.rand(20))  # y_test
        ]

        X_train, X_val, X_test, y_train, y_val, y_test = load_split_data('regression')

        self.assertEqual(X_train.shape, (100, 5))
        self.assertEqual(X_val.shape, (20, 5))
        self.assertEqual(X_test.shape, (20, 5))
        self.assertEqual(y_train.shape, (100,))
        self.assertEqual(y_val.shape, (20,))
        self.assertEqual(y_test.shape, (20,))

    def test_define_models_and_params(self):
        models, param_grids = define_models_and_params()
        self.assertIsInstance(models['Ridge'], Ridge)
        self.assertIn('alpha', param_grids['Ridge'])

    def test_define_classification_models_and_params(self):
        models, param_grids = define_classification_models_and_params()
        self.assertIn('LogisticRegression', models)
        self.assertIn('C', param_grids['LogisticRegression'])

    @patch("src.train.GridSearchCV")
    def test_perform_grid_search(self, mock_grid_search):
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.rand(100))

        # Mock the GridSearchCV behavior
        mock_grid_search_instance = MagicMock()
        mock_grid_search_instance.best_estimator_ = Ridge()
        mock_grid_search_instance.best_params_ = {'alpha': 1.0}
        mock_grid_search.return_value = mock_grid_search_instance

        models, param_grids = define_models_and_params()
        best_models = perform_grid_search(models, param_grids, X_train, y_train)

        self.assertIn('Ridge', best_models)
        self.assertIsInstance(best_models['Ridge'], Ridge)
        self.assertEqual(mock_grid_search_instance.fit.call_count, len(models))

    @patch("src.train.GridSearchCV")
    def test_perform_classification_grid_search(self, mock_grid_search):
        X_train = pd.DataFrame(np.random.rand(100, 5))
        y_train = pd.Series(np.random.randint(0, 2, 100))

        # Mock the GridSearchCV behavior
        mock_grid_search_instance = MagicMock()
        mock_grid_search_instance.best_estimator_ = RandomForestClassifier()
        mock_grid_search_instance.best_params_ = {'n_estimators': 100}
        mock_grid_search.return_value = mock_grid_search_instance

        models, param_grids = define_classification_models_and_params()
        best_models = perform_classification_grid_search(models, param_grids, X_train, y_train)

        self.assertIn('LogisticRegression', best_models)
        self.assertEqual(mock_grid_search_instance.fit.call_count, len(models))

    def test_evaluate_models(self):
        X_val = pd.DataFrame(np.random.rand(20, 5))
        y_val = pd.Series(np.random.rand(20))

        models = {'RandomForest': RandomForestRegressor()}
        models['RandomForest'].fit(X_val, y_val)

        results = evaluate_models(models, X_val, y_val)

        self.assertIn('RandomForest', results)
        self.assertIn('MSE', results['RandomForest'])

    def test_evaluate_classification_models(self):
        X_val = pd.DataFrame(np.random.rand(20, 5))
        y_val = pd.Series(np.random.randint(0, 2, 20))

        models = {'RandomForest': RandomForestClassifier()}
        models['RandomForest'].fit(X_val, y_val)

        results = evaluate_classification_models(models, X_val, y_val)

        self.assertIn('RandomForest', results)
        self.assertIn('Accuracy', results['RandomForest'])

    @patch("src.train.joblib.dump")
    def test_save_best_model(self, mock_dump):
        model = Ridge()
        folder_path = './models'
        model_name = 'ridge'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        save_best_model(model, model_name, folder_path)
        mock_dump.assert_called_once_with(model, os.path.join(folder_path, f'{model_name}_model.pkl'))


if __name__ == "__main__":
    unittest.main()