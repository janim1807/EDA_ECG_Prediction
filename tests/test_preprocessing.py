import unittest
import pandas as pd
from src.preprocessing import add_gaussian_noise, load_data, preprocess_data, bin_target_variable, \
    custom_train_test_split, save_splits
import os

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.sample_data = pd.DataFrame({
            'Participant': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
            'Group': ['G1', 'G2', 'G1', 'G2', 'G1', 'G2', 'G1', 'G2', 'G1', 'G2'],
            'Type': ['T1', 'T1', 'T2', 'T2', 'T1', 'T1', 'T2', 'T2', 'T1', 'T2'],
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'profit_loss_cum': [10, 20, -10, -20, 30, -30, 40, -40, 50, -50]
        })

        # Create directories if they don't exist
        os.makedirs('../data/modeling/train', exist_ok=True)
        os.makedirs('../data/modeling/dev', exist_ok=True)
        os.makedirs('../data/modeling/test', exist_ok=True)

    def test_load_data(self):
        file_path = 'sample.csv'
        df = pd.DataFrame({
            'Session': [1, 2],
            'Participant': ['A', 'B'],
            'Group': ['G1', 'G2'],
            'Type': ['T1', 'T2'],
            'feature1': [1, 2],
            'profit_loss_cum': [10, 20]
        })
        df.to_csv(file_path, index=False)

        # Test the load_data function
        loaded_df = load_data(file_path)
        self.assertNotIn('Session', loaded_df.columns)
        self.assertEqual(loaded_df.shape, (2, 5))
        os.remove(file_path)

    def test_preprocess_data(self):
        X, y, label_encoders = preprocess_data(self.sample_data)
        self.assertEqual(X.shape, (10, 5))
        self.assertEqual(y.shape, (10,))
        self.assertIn('Participant', label_encoders)
        self.assertIn('Group', label_encoders)
        self.assertIn('Type', label_encoders)

    def test_bin_target_variable(self):
        y = self.sample_data['profit_loss_cum']
        y_binned = bin_target_variable(y)
        self.assertEqual(y_binned.nunique(), 4)
        self.assertTrue(all(isinstance(x, int) for x in y_binned))

    def test_custom_train_test_split(self):
        X, y, _ = preprocess_data(self.sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(X, y, 'Participant', test_size=0.2, val_size=0.2)
        self.assertEqual(len(set(X_train['Participant']).intersection(set(X_test['Participant']))), 0)
        self.assertEqual(len(set(X_val['Participant']).intersection(set(X_test['Participant']))), 0)
        self.assertEqual(len(set(X_train['Participant']).intersection(set(X_val['Participant']))), 0)

    def test_save_splits(self):
        X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(self.sample_data.drop(columns=['profit_loss_cum']), self.sample_data['profit_loss_cum'], 'Participant', test_size=0.2, val_size=0.2)
        save_splits(X_train, X_val, X_test, y_train, y_val, y_test, 'test')

        # Check if files are created
        self.assertTrue(os.path.exists('../data/modeling/train/test_X_train.csv'))
        self.assertTrue(os.path.exists('../data/modeling/dev/test_X_val.csv'))
        self.assertTrue(os.path.exists('../data/modeling/test/test_X_test.csv'))
        self.assertTrue(os.path.exists('../data/modeling/train/test_y_train.csv'))
        self.assertTrue(os.path.exists('../data/modeling/dev/test_y_val.csv'))
        self.assertTrue(os.path.exists('../data/modeling/test/test_y_test.csv'))

        # Clean up
        os.remove('../data/modeling/train/test_X_train.csv')
        os.remove('../data/modeling/dev/test_X_val.csv')
        os.remove('../data/modeling/test/test_X_test.csv')
        os.remove('../data/modeling/train/test_y_train.csv')
        os.remove('../data/modeling/dev/test_y_val.csv')
        os.remove('../data/modeling/test/test_y_test.csv')

    def test_add_gaussian_noise(self):
        numerical_columns = ['feature1', 'feature2']
        noisy_data = add_gaussian_noise(self.sample_data.copy(), numerical_columns)
        for col in numerical_columns:
            self.assertFalse(noisy_data[col].equals(self.sample_data[col]))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()