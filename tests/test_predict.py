import warnings

warnings.filterwarnings('ignore')

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score

from src.predict import (load_test_data, load_model, evaluate_regression_model, evaluate_classification_model)


class TestPredictPipeline(unittest.TestCase):

    @patch("src.predict.pd.read_csv")
    def test_load_test_data(self, mock_read_csv):
        # Setup mock return values
        mock_read_csv.side_effect = [
            pd.DataFrame(np.random.rand(20, 5)),  # X_test
            pd.Series(np.random.rand(20))  # y_test
        ]

        X_test, y_test = load_test_data('regression')

        self.assertEqual(X_test.shape, (20, 5))
        self.assertEqual(y_test.shape, (20,))

    @patch("src.predict.joblib.load")
    def test_load_model(self, mock_load):
        # Mock the return value of joblib.load
        mock_load.return_value = MagicMock()

        model_path = '../models/sample_model.pkl'
        model = load_model(model_path)

        mock_load.assert_called_once_with(model_path)
        self.assertIsInstance(model, MagicMock)

    def test_evaluate_regression_model(self):
        model = MagicMock()
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = pd.Series(np.random.rand(20))
        y_pred = np.random.rand(20)

        model.predict.return_value = y_pred

        with patch('builtins.print') as mock_print:
            evaluate_regression_model(model, X_test, y_test)
            mock_print.assert_called_with(
                f"Regression Test Metrics: MSE: {mean_squared_error(y_test, y_pred):.4f}, "
                f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}, "
                f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, "
                f"R2: {r2_score(y_test, y_pred):.4f}"
            )

    def test_evaluate_classification_model(self):
        model = MagicMock()
        X_test = pd.DataFrame(np.random.rand(20, 5))
        y_test = pd.Series(np.random.randint(0, 2, 20))
        y_pred = np.random.randint(0, 2, 20)

        model.predict.return_value = y_pred
        model.predict_proba.return_value = np.random.rand(20, 2)

        with patch('builtins.print') as mock_print:
            evaluate_classification_model(model, X_test, y_test)
            mock_print.assert_called()

    @patch("src.predict.load_test_data")
    @patch("src.predict.load_model")
    @patch("src.predict.evaluate_regression_model")
    @patch("src.predict.evaluate_classification_model")
    def test_main(self, mock_eval_class, mock_eval_reg, mock_load_model, mock_load_test_data):
        # Mock the load_test_data function to return dummy data with a 'Participant' column
        mock_load_test_data.side_effect = [
            (pd.DataFrame(np.random.rand(20, 5), columns=[f'feature_{i}' for i in range(5)]).assign(Participant=1),
             pd.Series(np.random.rand(20))),  # Regression data
            (pd.DataFrame(np.random.rand(20, 5), columns=[f'feature_{i}' for i in range(5)]).assign(Participant=1),
             pd.Series(np.random.randint(0, 2, 20)))  # Classification data
        ]

        # Mock the load_model function to return a dummy model
        mock_load_model.return_value = MagicMock()

        # Call the main function
        from src.predict import main
        main()

        # Check that the data loading functions were called
        mock_load_test_data.assert_any_call('regression_augmented')
        mock_load_test_data.assert_any_call('categorical_augmented')

        # Check that the model loading functions were called
        mock_load_model.assert_any_call('../models/XGBRegressor_model.pkl')
        mock_load_model.assert_any_call('../models/KNeighbors_model.pkl')

        # Check that the evaluation functions were called
        mock_eval_reg.assert_called_once()
        mock_eval_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()