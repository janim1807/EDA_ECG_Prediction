import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.additional_features import (load_data_from_folders, preprocess_data, calculate_relative_changes,
                                     count_windows_per_participant, calculate_wallet_history, assign_windows,
                                     calculate_cumulative_profit_loss, create_sliding_windows,
                                     calculate_sliding_window_profit_loss, verify_window_counts,
                                     remove_inconsistent_windows,
                                     add_profit_loss_to_df, add_profit_loss_to_df_sliding_windows, clean_data, main)


class TestAdditionalFeatures(unittest.TestCase):

    @patch('src.additional_features.os.listdir')
    @patch('src.additional_features.pd.read_csv')
    def test_load_data_from_folders(self, mock_read_csv, mock_listdir):
        # weird error that it fails, but this is hard to test aswell
        mock_listdir.return_value = ['all_apps_wide_1.csv', 'ecg_results.csv', 'eda_results.csv']
        mock_read_csv.return_value = pd.DataFrame(np.random.rand(10, 5))

        participants, ecg, eda = load_data_from_folders('all_apps_wide', 'ecg_results.csv', 'eda_results.csv')

        self.assertEqual(len(participants), 0)
        self.assertEqual(len(ecg), 0)
        self.assertEqual(len(eda), 0)

    def test_preprocess_data(self):
        df_participants = pd.DataFrame({'participant.code': ['p1', 'p2', 'p3'],
                                        'stockmarket.1.player.cash': [100, 200, 300]})
        df_ecg = pd.DataFrame({'Participant': ['p1', 'p2', 'p3'], 'ecg1': [1, 2, 3]})
        df_eda = pd.DataFrame(
            {'Participant': ['p1', 'p2', 'p3'], 'eda1': [1, 2, 3], 'Session_Time': ['t1', 't2', 't3']})

        df_participants, df_ecg, df_eda = preprocess_data(df_participants, df_ecg, df_eda, 'p1')

        self.assertNotIn('p1', df_participants['participant.code'].values)
        self.assertNotIn('p1', df_eda['Participant'].values)
        self.assertNotIn('Session_Time', df_eda.columns)

    def test_calculate_relative_changes(self):
        df_combined = pd.DataFrame({'Participant': ['p1', 'p1', 'p2', 'p2'],
                                    'Type': ['Baseline', 'Session1', 'Baseline', 'Session1'],
                                    'ecg1': [1, 2, 1, 2], 'eda1': [1, 2, 1, 2]})
        df_relative = calculate_relative_changes(df_combined)

        self.assertEqual(df_relative.loc[1, 'ecg1'], 100.0)
        self.assertEqual(df_relative.loc[1, 'eda1'], 100.0)

    def test_count_windows_per_participant(self):
        df_combined = pd.DataFrame({'Participant': ['p1', 'p1', 'p2', 'p2'],
                                    'Type': ['Baseline', 'Window 1', 'Baseline', 'Window 1']})
        window_counts = count_windows_per_participant(df_combined)

        self.assertEqual(window_counts['p1'], 1)
        self.assertEqual(window_counts['p2'], 1)

    def test_calculate_wallet_history(self):
        df_participants = pd.DataFrame({'participant.id_in_session': [1, 2], 'participant.code': ['p1', 'p2'],
                                        'stockmarket.1.player.cash': [100, 150],
                                        'stockmarket.1.player.dividend': [10, 15],
                                        'stockmarket.1.player.interest': [5, 10], 'stockmarket.1.player.stocks': [2, 3],
                                        'stockmarket.1.player.d1_start': [1609459200000, 1609459200000],
                                        })
        data, discrepancies = calculate_wallet_history(df_participants)

        self.assertEqual(len(data), 2)
        self.assertEqual(len(discrepancies), 0)

    def test_assign_windows(self):
        data = [{'participant_code': 'p1', 'profit_loss_history': [1, 2, 3], 'start_times': ['2021-01-01 00:00:00'] * 3,
                 'end_times': ['2021-01-01 00:03:00'] * 3}]
        participant_window_counts = {'p1': 2}
        assigned_windows = assign_windows(data, participant_window_counts)

        self.assertEqual(len(assigned_windows[0]['window_assignments']), 1)

    def test_calculate_cumulative_profit_loss(self):
        data = [{'participant_id': 3, 'participant_code': 'zzq192rp',
                 'wallet_history': [140, 136.0, 137.17],
                 'profit_loss_history': [0, -4.0, 1.17],
                 'start_times': ['2024-05-07 14:18:33', '2024-05-07 14:19:22', '2024-05-07 14:20:23'],
                 'end_times': ['2024-05-07 14:19:22', '2024-05-07 14:20:23', '2024-05-07 14:21:17',],
                 'window_assignments': [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]}]

        calculated_data = calculate_cumulative_profit_loss(data)

        self.assertEqual(calculated_data[0]['profit_loss_cum_per_window'],
                         [-2.83, 0, 0])

    def test_create_sliding_windows(self):
        window_assignments = [[0, 1, 2], [3, 4, 5]]
        original, left_shifted, right_shifted = create_sliding_windows(window_assignments)

        self.assertEqual(left_shifted[0], [0, 1])
        self.assertEqual(right_shifted[-1], [4, 5])

    def test_calculate_sliding_window_profit_loss(self):
        data = [{'participant_code': 'p1', 'profit_loss_history': [0, 1, 2, 3],
                 'window_assignments': [[0], [1, 2], [3]]}]
        calculated_data = calculate_sliding_window_profit_loss(data)

        self.assertIn('original', calculated_data[0]['profit_loss_cum_per_window'])
        self.assertIn('left', calculated_data[0]['profit_loss_cum_per_window'])
        self.assertIn('right', calculated_data[0]['profit_loss_cum_per_window'])

    def test_verify_window_counts(self):
        data = [{'participant_code': 'p1', 'window_assignments': [[0], [1, 2]]}]
        df_combined = pd.DataFrame({'Participant': ['p1', 'p1'], 'Type': ['Window 1', 'Window 2']})
        inconsistent_participants = verify_window_counts(data, df_combined)

        self.assertEqual(len(inconsistent_participants), 1)

    def test_remove_inconsistent_windows(self):
        df_combined = pd.DataFrame({'Participant': ['p1', 'p1', 'p2'], 'Type': ['Window 1', 'Window 2', 'Window 1']})
        inconsistent_participants = [('p1', 2, 1)]
        df_cleaned = remove_inconsistent_windows(df_combined, inconsistent_participants)

        self.assertEqual(len(df_cleaned), 2)

    def test_add_profit_loss_to_df(self):
        data = [{'participant_code': 'p1', 'profit_loss_cum_per_window': [100, 200]}]
        df_combined = pd.DataFrame({'Participant': ['p1', 'p1'], 'Type': ['Window 1', 'Window 2']})
        df_combined = add_profit_loss_to_df(data, df_combined)

        self.assertEqual(df_combined.loc[0, 'profit_loss_cum'], 100)
        self.assertEqual(df_combined.loc[1, 'profit_loss_cum'], 200)

    def test_add_profit_loss_to_df_sliding_windows(self):
        data = [{'participant_code': 'p1',
                 'profit_loss_cum_per_window': {'original': [100, 200], 'left': [90, 190], 'right': [110, 210]},
                 'window_assignments': {'original': [[0], [1]], 'left': [[0], [1]], 'right': [[0], [1]]}}]
        df_combined = pd.DataFrame({'Participant': ['p1', 'p1'], 'Type': ['Window 1', 'Window 2']})
        new_rows_df = add_profit_loss_to_df_sliding_windows(data, df_combined)

        self.assertTrue('profit_loss_cum' in new_rows_df.columns)

    def test_clean_data(self):
        df_combined = pd.DataFrame({'Participant': ['p1', 'p2'], 'Type': ['Window 1', 'Window 2'], 'col1': [1, np.nan]})
        df_cleaned = clean_data(df_combined)

        self.assertNotIn('col1', df_cleaned.columns)

    @patch("builtins.print")
    @patch("src.additional_features.load_data_from_folders")
    @patch("src.additional_features.preprocess_data")
    @patch("src.additional_features.calculate_wallet_history")
    @patch("src.additional_features.calculate_relative_changes")
    @patch("src.additional_features.count_windows_per_participant")
    @patch("src.additional_features.verify_window_counts")
    @patch("src.additional_features.remove_inconsistent_windows")
    @patch("src.additional_features.add_profit_loss_to_df")
    @patch("src.additional_features.add_profit_loss_to_df_sliding_windows")
    @patch("src.additional_features.clean_data")
    def test_main(self, mock_clean_data, mock_add_profit_loss_to_df_sliding_windows, mock_add_profit_loss_to_df,
                  mock_remove_inconsistent_windows, mock_verify_window_counts, mock_count_windows_per_participant,
                  mock_calculate_relative_changes, mock_calculate_wallet_history, mock_preprocess_data,
                  mock_load_data_from_folders, mock_print):
        # To pass this test everything need to be mocked, that would be too much.
        mock_participants_df = pd.DataFrame({
            'participant.code': ['p1', 'p2'],
            'participant.id_in_session': [1, 2],
            'stockmarket.1.player.cash': [100, 200],
            'stockmarket.1.player.dividend': [10, 20],
            'stockmarket.1.player.interest': [5, 15],
            'stockmarket.1.player.stocks': [1, 2],
            'stockmarket.2.player.avg_price': [50, 60],
            'stockmarket.40.player.final_score': [1000, 1200],
            'stockmarket.1.player.d1_start': [1609459200000, 1609459200000],
            'stockmarket.2.player.d1_start': [1609459260000, 1609459260000]
        })

        mock_ecg_df = pd.DataFrame({
            'Participant': ['p1', 'p2'],
            'Type': ['Window 1', 'Window 2'],
            'Group': ['G1', 'G2'],
            'ecg1': [0.1, 0.2]
        })

        mock_eda_df = pd.DataFrame({
            'Participant': ['p1', 'p2'],
            'Type': ['Window 1', 'Window 2'],
            'Group': ['G1', 'G2'],
            'eda1': [0.3, 0.4],
            'Session_Time': ['2024-05-07 14:18:33', '2024-05-07 14:18:33']
        })

        mock_combined_df = pd.DataFrame({
            'Participant': ['p1', 'p1', 'p2', 'p2'],
            'Type': ['Baseline', 'Window 1', 'Baseline', 'Window 1'],
            'Group': ['G1', 'G1', 'G2', 'G2'],
            'ecg1': [0.1, 0.2, 0.1, 0.2],
            'eda1': [0.3, 0.4, 0.3, 0.4]
        })

        mock_cleaned_df = pd.DataFrame({
            'Participant': ['p1', 'p2'],
            'Group': ['G1', 'G2'],
            'Type': ['Window 1', 'Window 2'],
            'profit_loss_cum': [100, 200]
        })

        mock_data = [{
            'participant_code': 'p1',
            'profit_loss_history': [0, 10],
            'start_times': ['2024-05-07 14:00:00', '2024-05-07 14:03:00'],
            'end_times': ['2024-05-07 14:03:00', '2024-05-07 14:06:00']
        }]

        mock_load_data_from_folders.return_value = (mock_participants_df, mock_ecg_df, mock_eda_df)
        mock_preprocess_data.return_value = (mock_participants_df, mock_ecg_df, mock_eda_df)
        mock_calculate_wallet_history.return_value = (mock_data, [])
        mock_calculate_relative_changes.return_value = mock_combined_df
        mock_count_windows_per_participant.return_value = {'p1': 1, 'p2': 1}
        mock_verify_window_counts.return_value = []
        mock_remove_inconsistent_windows.return_value = mock_combined_df
        mock_add_profit_loss_to_df.return_value = mock_combined_df
        mock_add_profit_loss_to_df_sliding_windows.return_value = mock_cleaned_df
        mock_clean_data.return_value = mock_cleaned_df

        main()

        mock_load_data_from_folders.assert_called_once()
        mock_preprocess_data.assert_called_once()
        mock_calculate_wallet_history.assert_called_once()
        mock_calculate_relative_changes.assert_called_once()
        mock_count_windows_per_participant.assert_called_once()
        mock_verify_window_counts.assert_called_once()
        mock_remove_inconsistent_windows.assert_called_once()
        mock_add_profit_loss_to_df.assert_called_once()
        mock_add_profit_loss_to_df_sliding_windows.assert_called_once()
        mock_clean_data.assert_called_once()

if __name__ == "__main__":
    unittest.main()
