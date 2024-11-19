import os
import pandas as pd
from datetime import datetime, timedelta
from copy import deepcopy


def load_data_from_folders(participant_file_prefix, ecg_file_name, eda_file_name):
    """
    Load data from multiple folders containing participant, ECG, and EDA data files.

    Args:
        participant_file_prefix (str): Prefix of participant data files.
        ecg_file_name (str): Name of the ECG data file.
        eda_file_name (str): Name of the EDA data file.

    Returns:
        tuple: Three DataFrames containing participant data, ECG data, and EDA data.
    """
    all_participants = []
    all_ecg = []
    all_eda = []
    folder_path = '../data/raw'
    folder_names = [os.path.join(folder_path, str(i)) for i in range(1, 8)]

    for folder in folder_names:
        participant_files = [f for f in os.listdir(folder) if f.startswith(participant_file_prefix)]
        for file in participant_files:
            df_participant = pd.read_csv(os.path.join(folder, file))
            all_participants.append(df_participant)

        # Load ECG data
        ecg_file_path = os.path.join(folder, ecg_file_name)
        if os.path.exists(ecg_file_path):
            df_ecg = pd.read_csv(ecg_file_path)
            all_ecg.append(df_ecg)

        # Load EDA data
        eda_file_path = os.path.join(folder, eda_file_name)
        if os.path.exists(eda_file_path):
            df_eda = pd.read_csv(eda_file_path)
            all_eda.append(df_eda)

    # Concatenate all dataframes, ensuring consistent columns
    df_participants_combined = pd.concat(all_participants, ignore_index=True).reindex(
        columns=all_participants[0].columns)
    df_ecg_combined = pd.concat(all_ecg, ignore_index=True).reindex(columns=all_ecg[0].columns)
    df_eda_combined = pd.concat(all_eda, ignore_index=True).reindex(columns=all_eda[0].columns)

    return df_participants_combined, df_ecg_combined, df_eda_combined


def preprocess_data(df_participants_combined, df_ecg_combined, df_eda_combined, participant_id):
    """
    Preprocess data by removing specified participant and filtering based on valid participant IDs.

    Args:
        df_participants_combined (pd.DataFrame): Combined participant data.
        df_ecg_combined (pd.DataFrame): Combined ECG data.
        df_eda_combined (pd.DataFrame): Combined EDA data.
        participant_id (str): Participant ID to remove.

    Returns:
        tuple: Preprocessed participant, ECG, and EDA DataFrames.
    """
    # Remove specified participant
    df_participants_combined = df_participants_combined[df_participants_combined['participant.code'] != participant_id]
    df_eda_combined = df_eda_combined[df_eda_combined['Participant'] != participant_id]

    # Get valid participant IDs
    participant_ids_ecg = set(df_ecg_combined['Participant'])
    participant_ids_eda = set(df_eda_combined['Participant'])
    valid_participant_ids = participant_ids_ecg.intersection(participant_ids_eda)
    df_participants_combined = df_participants_combined[
        df_participants_combined['participant.code'].isin(valid_participant_ids)]

    df_eda_combined.drop(columns=['Session_Time'], inplace=True)

    return df_participants_combined, df_ecg_combined, df_eda_combined


def calculate_relative_changes(df_combined):
    """
    Calculate relative changes from baseline for each participant.

    Args:
        df_combined (pd.DataFrame): Combined DataFrame with ECG and EDA data.

    Returns:
        pd.DataFrame: DataFrame with relative changes.
    """
    relevant_columns = [col for col in df_combined.columns if col not in ['Session', 'Participant', 'Group', 'Type']]
    df_combined_relative = df_combined.copy()

    participants = df_combined['Participant'].unique()
    for participant in participants:
        baseline_values = None
        for idx, row in df_combined[df_combined['Participant'] == participant].iterrows():
            if row['Type'] == 'Baseline':
                baseline_values = row[relevant_columns]
            else:
                if baseline_values is not None:
                    for col in relevant_columns:
                        baseline_value = baseline_values[col]
                        if pd.notna(baseline_value) and baseline_value != 0 and pd.notna(row[col]):
                            df_combined_relative.loc[idx, col] = ((row[col] - baseline_value) / baseline_value) * 100
                        else:
                            df_combined_relative.loc[idx, col] = row[col]
    return df_combined_relative


def count_windows_per_participant(df_combined):
    """
    Count the number of windows (Type values) for each participant.

    Args:
        df_combined (pd.DataFrame): Combined DataFrame with relative changes.

    Returns:
        dict: Dictionary with participant ID as key and window count as value.
    """
    participant_window_counts = {}
    for participant_id in df_combined['Participant'].unique():
        participant_data = df_combined[df_combined['Participant'] == participant_id]
        first_baseline_found = False
        window_count = 0
        for index, row in participant_data.iterrows():
            if row['Type'] == "Baseline":
                if not first_baseline_found:
                    first_baseline_found = True
                else:
                    break
            elif first_baseline_found:
                window_count += 1
        participant_window_counts[participant_id] = window_count
    return participant_window_counts


def calculate_wallet_history(df_participants_combined):
    """
    Calculate wallet history and profit/loss for each participant.

    Args:
        df_participants_combined (pd.DataFrame): Combined participant data.

    Returns:
        list: List of dictionaries with wallet history and profit/loss for each participant.
    """
    data = []
    discrepancies = []

    for index, row in df_participants_combined.iterrows():
        wallet_history = [140]
        start_times = []
        end_times = []
        profit_loss_history = [0]
        for i in range(1, 41):
            cash = row[f'stockmarket.{i}.player.cash']
            dividend = row[f'stockmarket.{i}.player.dividend']
            interest = row[f'stockmarket.{i}.player.interest']
            stocks = row[f'stockmarket.{i}.player.stocks']
            next_avg_price = row[f'stockmarket.{i + 1}.player.avg_price'] if i < 40 else None
            wallet = cash + dividend + interest + stocks * next_avg_price if next_avg_price is not None else row[
                'stockmarket.40.player.final_score']
            wallet_history.append(wallet)
            start_time_col = f'stockmarket.{i}.player.d1_start'
            end_time_col = f'stockmarket.{i + 1}.player.d1_start' if i < 40 else None
            start_time = datetime.fromtimestamp(row[start_time_col] / 1000.0).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(
                row[start_time_col]) else None
            end_time = datetime.fromtimestamp(row[end_time_col] / 1000.0).strftime(
                '%Y-%m-%d %H:%M:%S') if end_time_col is not None and pd.notna(row[end_time_col]) else None
            start_times.append(start_time)
            end_times.append(end_time)
            profit_loss = wallet_history[i] - wallet_history[i - 1]
            profit_loss_history.append(profit_loss)
        calculated_final_wallet = wallet_history[-1]
        reported_final_score = row['stockmarket.40.player.final_score']
        if calculated_final_wallet != reported_final_score:
            discrepancies.append({
                'participant_id': row['participant.id_in_session'],
                'participant_code': row['participant.code'],
                'calculated_final_wallet': calculated_final_wallet,
                'reported_final_score': reported_final_score
            })
        data.append({'participant_id': row['participant.id_in_session'], 'participant_code': row['participant.code'],
                     'wallet_history': wallet_history, 'profit_loss_history': profit_loss_history,
                     'start_times': start_times, 'end_times': end_times})
    return data, discrepancies


def assign_windows(data, participant_window_counts):
    """
    Assign rounds to windows based on time duration.

    Args:
        data (list): List of dictionaries with wallet history and profit/loss for each participant.
        participant_window_counts (dict): Dictionary with participant ID as key and window count as value.

    Returns:
        list: Updated list of dictionaries with window assignments.
    """
    window_duration = timedelta(minutes=3)

    for entry in data:
        participant_id = entry['participant_code']
        profit_loss_history = entry['profit_loss_history']
        start_times_str = entry['start_times']
        end_times_str = entry['end_times']

        window_count = participant_window_counts.get(participant_id, 0)

        start_times = [datetime.strptime(time, '%Y-%m-%d %H:%M:%S') for time in start_times_str if time]
        end_times = [datetime.strptime(time, '%Y-%m-%d %H:%M:%S') for time in end_times_str if time]

        window_assignments = [[] for _ in range(window_count)]
        current_window_index = 0
        current_window_start = start_times[0]
        current_window_end = current_window_start + window_duration
        last_window_end = current_window_end
        round_end_time = start_times[0]
        last_round_end_time = start_times[0]

        for round_index in range(len(start_times_str) - 1):
            last_round_end_time = round_end_time
            round_end_time = end_times[round_index]

            while current_window_index < window_count and round_end_time >= current_window_end:
                current_window_index += 1
                if current_window_index < window_count:
                    current_window_start = last_window_end
                    current_window_end = current_window_start + window_duration
                    last_window_end = current_window_end

            if current_window_index < window_count:
                window_assignments[current_window_index].append(round_index)

        window_assignments = [window for window in window_assignments if window]
        entry['window_assignments'] = window_assignments

    return data


def calculate_cumulative_profit_loss(data):
    """
       Calculate the cumulative profit/loss for each window in the provided data.

       Parameters:
       data (list of dict): Each dict contains 'profit_loss_history' (list of float) and
                            'window_assignments' (list of list of int).

       Returns:
       list of dict: The input data with an additional key 'profit_loss_cum_per_window'
                     for each entry, containing cumulative profit/loss per window.
       """
    for entry in data:
        profit_loss_history = entry['profit_loss_history']
        window_assignments = entry['window_assignments']
        profit_loss_cum_per_window = []

        for window_indices in window_assignments:
            cumulative_profit_loss = 0
            for round_index in window_indices:
                try:
                    cumulative_profit_loss += profit_loss_history[round_index + 1]
                except IndexError:
                    cumulative_profit_loss += 0  # Default to 0 if index is out of range
            profit_loss_cum_per_window.append(cumulative_profit_loss)

        entry['profit_loss_cum_per_window'] = profit_loss_cum_per_window

    return data


def create_sliding_windows(window_assignments):
    """
    Create original, left-shifted, and right-shifted sliding windows from the given window assignments.

    Args:
        window_assignments (list): List of window assignments for each participant.

    Returns:
        tuple: Three lists containing original windows, left-shifted windows, and right-shifted windows.
    """
    original_windows = deepcopy(window_assignments)  # Deep copy of the input data

    left_shifted_windows = []
    if len(original_windows) > 0:
        left_shifted_windows.append(original_windows[0][:-1])

    for i in range(1, len(original_windows)):
        current_window = [original_windows[i - 1][-1]] + original_windows[i][:-1]
        left_shifted_windows.append(current_window)

    right_shifted_windows = []

    for i in range(1, len(original_windows)):
        prev_window_end = original_windows[i - 1][1:] + [original_windows[i][0]]
        current_window = original_windows[i][1:]
        right_shifted_windows.append(prev_window_end)

    right_shifted_windows.append(original_windows[-1][1:])

    return original_windows, left_shifted_windows, right_shifted_windows


def calculate_sliding_window_profit_loss(data):
    """
    Calculate cumulative profit/loss per window for original, left-shifted, and right-shifted windows.

    Args:
        data (list): List of dictionaries with wallet history and profit/loss for each participant.

    Returns:
        list: Updated list of dictionaries with cumulative profit/loss per window for sliding windows.
    """
    data_sliding_windows = deepcopy(data)

    for entry in data_sliding_windows:
        original_windows, left_shifted_windows, right_shifted_windows = create_sliding_windows(
            entry['window_assignments'])

        entry['window_assignments'] = {
            'original': original_windows,
            'left': left_shifted_windows,
            'right': right_shifted_windows
        }

    for entry in data_sliding_windows:
        profit_loss_history = entry['profit_loss_history']
        window_assignments = entry['window_assignments']
        profit_loss_cum_per_window = {
            'original': [],
            'left': [],
            'right': []
        }
        for key, values in window_assignments.items():
            for window_indices in values:
                if not window_indices:
                    profit_loss_cum_per_window[key].append(None)
                    continue
                cumulative_profit_loss = 0
                for round_index in window_indices:
                    if round_index + 1 < len(profit_loss_history):
                        cumulative_profit_loss += profit_loss_history[round_index + 1]
                profit_loss_cum_per_window[key].append(cumulative_profit_loss)
        entry['profit_loss_cum_per_window'] = profit_loss_cum_per_window

    return data_sliding_windows



def verify_window_counts(data, df_combined):
    """
    Verify the window counts in the data variable against the df_combined DataFrame.

    Args:
        data (list): List of participant data entries.
        df_combined (pd.DataFrame): Combined DataFrame containing participant data.

    Returns:
        list: List of inconsistent participants with their respective window counts.
    """
    participant_window_counts = {}

    for entry in data:
        participant_id = entry['participant_code']
        window_count = len(entry['window_assignments'])
        participant_window_counts[participant_id] = window_count

    df_window_counts = df_combined.groupby('Participant')['Type'].nunique() - 1
    inconsistent_participants = []

    for participant, window_count in participant_window_counts.items():
        if df_window_counts.get(participant, 0) != window_count:
            inconsistent_participants.append((participant, df_window_counts.get(participant, 0), window_count))

    return inconsistent_participants


def remove_inconsistent_windows(df_combined, inconsistent_participants):
    """
    Remove inconsistent windows from df_combined based on window counts.

    Args:
        df_combined (pd.DataFrame): Combined DataFrame containing participant data.
        inconsistent_participants (list): List of inconsistent participants with their respective window counts.

    Returns:
        pd.DataFrame: DataFrame with inconsistent windows removed.
    """
    for participant, df_window_count, data_window_count in inconsistent_participants:
        window_to_remove = f'Window {df_window_count}'
        df_combined = df_combined[
            ~((df_combined['Participant'] == participant) & (df_combined['Type'] == window_to_remove))]
    return df_combined


def add_profit_loss_to_df(data, df_combined):
    """
    Add the profit_loss_cum_per_window data to the corresponding windows in df_combined.

    Args:
        data (list): List of participant data entries.
        df_combined (pd.DataFrame): Combined DataFrame containing participant data.

    Returns:
        pd.DataFrame: Updated DataFrame with the cumulative profit/loss data added.
    """
    profit_loss_dict = {}
    for entry in data:
        participant_id = entry['participant_code']
        profit_loss_cum_per_window = entry['profit_loss_cum_per_window']
        profit_loss_dict[participant_id] = profit_loss_cum_per_window
    df_combined['profit_loss_cum'] = None

    for index, row in df_combined.iterrows():
        participant_id = row['Participant']
        window_type = row['Type']

        if participant_id in profit_loss_dict:
            window_number = int(window_type.split()[-1]) - 1
            if window_number < len(profit_loss_dict[participant_id]):
                df_combined.at[index, 'profit_loss_cum'] = profit_loss_dict[participant_id][window_number]
    return df_combined


def add_profit_loss_to_df_sliding_windows(data, df_combined):
    """
    Add the profit_loss_cum_per_window data for original, left-shifted, and right-shifted windows to df_combined.

    Args:
        data (list): List of participant data entries.
        df_combined (pd.DataFrame): Combined DataFrame containing participant data.

    Returns:
        pd.DataFrame: Updated DataFrame with the cumulative profit/loss data added for sliding windows.
    """
    profit_loss_dict = {}
    window_length_dict = {}
    for entry in data:
        participant_id = entry['participant_code']
        profit_loss_cum_per_window_original = entry['profit_loss_cum_per_window']['original']
        profit_loss_cum_per_window_left = entry['profit_loss_cum_per_window']['left']
        profit_loss_cum_per_window_right = entry['profit_loss_cum_per_window']['right']
        profit_loss_dict[participant_id] = {'original': profit_loss_cum_per_window_original,
                                            'left': profit_loss_cum_per_window_left,
                                            'right': profit_loss_cum_per_window_right}
        window_length = []
        for window in entry['window_assignments']['original']:
            window_length.append(len(window))
        window_length_dict[participant_id] = window_length
    df_combined['profit_loss_cum'] = None
    new_rows = []
    previous_window_number = 0
    participant_previous_row = ''
    participant_next_row = ''
    for index, row in df_combined.iterrows():
        participant_id = df_combined.iloc[index]['Participant']
        if index + 2 < len(df_combined):
            participant_next_row = df_combined.iloc[index + 1]['Participant']
        else:
            participant_next_row = ''
        window_type = row['Type']

        if participant_id in profit_loss_dict:
            window_number = int(window_type.split()[-1]) - 1
            if window_number <= (len(profit_loss_dict[participant_id]['original']) - 1):
                df_combined.at[index, 'profit_loss_cum'] = profit_loss_dict[participant_id]['original'][window_number]
                value_links = profit_loss_dict[participant_id]['left'][window_number]
                value_rechts = profit_loss_dict[participant_id]['right'][window_number]

                new_row_left = row.copy()
                new_row_left['profit_loss_cum'] = value_links
                new_row_right = row.copy()
                new_row_right['profit_loss_cum'] = value_rechts

                if index > 0:
                    previous_window_type = df_combined.iloc[index - 1]['Type']
                    previous_window_number = int(previous_window_type.split()[-1]) - 1
                else:
                    previous_window_number = None

                if index < len(df_combined) - 1:
                    next_window_type = df_combined.iloc[index + 1]['Type']
                    next_window_number = int(next_window_type.split()[-1]) - 1
                else:
                    next_window_number = None

                exclude_cols = ['Session', 'Participant', 'Group', 'Type', 'profit_loss_cum']

                for col in df_combined.columns:
                    if col not in exclude_cols:
                        previous_row_value = df_combined[col].shift(1).iloc[index]
                        current_row_value = df_combined[col].iloc[index]
                        next_row_value = df_combined[col].shift(-1).iloc[index]
                        if current_row_value != None:
                            if participant_previous_row != participant_id:
                                new_row_left[col] = current_row_value
                                new_row_right[col] = (
                                        (next_row_value / window_length_dict[participant_id][next_window_number]
                                         + (window_length_dict[participant_id][
                                                window_number] - 1) * current_row_value /
                                         window_length_dict[participant_id][window_number])
                                        / (1 / window_length_dict[participant_id][next_window_number] + (
                                        window_length_dict[participant_id][window_number] - 1) /
                                           window_length_dict[participant_id][window_number]))
                            elif participant_previous_row == participant_id:
                                new_row_left[col] = ((previous_row_value / window_length_dict[participant_id][
                                    previous_window_number]
                                                      + (window_length_dict[participant_id][
                                                             window_number] - 1) * current_row_value /
                                                      window_length_dict[participant_id][window_number])
                                                     / (1 / window_length_dict[participant_id][
                                            previous_window_number] + (window_length_dict[participant_id][
                                                                           window_number] - 1) /
                                                        window_length_dict[participant_id][window_number]))
                                if participant_next_row == participant_id:
                                    new_row_right[col] = (
                                            (next_row_value / window_length_dict[participant_id][next_window_number]
                                             + (window_length_dict[participant_id][
                                                    window_number] - 1) * current_row_value /
                                             window_length_dict[participant_id][window_number])
                                            / (1 / window_length_dict[participant_id][next_window_number] + (
                                            window_length_dict[participant_id][window_number] - 1) /
                                               window_length_dict[participant_id][window_number]))
                                else:
                                    new_row_right[col] = current_row_value

                new_rows.append(new_row_left)
                new_rows.append(new_row_right)
        participant_previous_row = participant_id

    new_rows_df = pd.DataFrame(new_rows)
    return new_rows_df


def clean_data(df_combined):
    """
    Remove columns with NaN values from df_combined.

    Args:
        df_combined (pd.DataFrame): Combined DataFrame containing participant data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with columns containing NaN values removed.
    """
    nan_counts = df_combined.isna().sum()
    nan_counts_df = pd.DataFrame(nan_counts, columns=['NaN Count'])
    columns_with_nan = nan_counts_df[nan_counts_df['NaN Count'] > 0].index
    df_combined_cleaned = df_combined.drop(columns=columns_with_nan)
    return df_combined_cleaned


def main():
    """
        Main function to control the creation of the new features.
       """
    participant_file_prefix = 'all_apps_wide'
    ecg_file_name = 'ecg_results.csv'
    eda_file_name = 'eda_results.csv'
    participant_id = 'ae2kja5u'

    df_participants_combined, df_ecg_combined, df_eda_combined = load_data_from_folders(
        participant_file_prefix, ecg_file_name, eda_file_name)

    df_participants_combined, df_ecg_combined, df_eda_combined = preprocess_data(
        df_participants_combined, df_ecg_combined, df_eda_combined, participant_id)

    df_combined = pd.merge(df_ecg_combined, df_eda_combined, on=['Participant', 'Type', 'Group'])
    df_combined = calculate_relative_changes(df_combined)

    participant_window_counts = count_windows_per_participant(df_combined)

    data, discrepancies = calculate_wallet_history(df_participants_combined)
    if discrepancies:
        print("\nDiscrepancies found between calculated final wallet and reported final score:")
        for discrepancy in discrepancies:
            print(discrepancy)
    else:
        print("\nNo discrepancies found between calculated final wallet and reported final score.")

    data = assign_windows(data, participant_window_counts)
    data = calculate_cumulative_profit_loss(data)

    data_sliding_windows = calculate_sliding_window_profit_loss(data)

    inconsistent_participants = verify_window_counts(data, df_combined)

    df_combined = df_combined[df_combined['Type'] != 'Baseline']
    add_profit_loss_to_df(data, df_combined)

    df_combined = remove_inconsistent_windows(df_combined, inconsistent_participants)
    inconsistent_participants_after_removal = verify_window_counts(data, df_combined)

    if inconsistent_participants_after_removal:
        print(f"Number of inconsistent participants after removal: {len(inconsistent_participants_after_removal)}")
        for participant, df_window_count, data_window_count in inconsistent_participants_after_removal:
            print(
                f"Inconsistent participant after removal: {participant}, df_window_count: {df_window_count}, "
                f"data_window_count: {data_window_count}")
    else:
        print("All participants have consistent window counts after removal.")

    df_combined.reset_index(drop=True, inplace=True)

    df_combined_sliding_windows = deepcopy(df_combined)
    new_rows_df = add_profit_loss_to_df_sliding_windows(data_sliding_windows, df_combined_sliding_windows)
    df_combined_sliding_windows = pd.concat([df_combined_sliding_windows, new_rows_df], ignore_index=True)

    df_combined_cleaned_sliding_windows = clean_data(df_combined_sliding_windows)

    df_combined_cleaned_sliding_windows.to_csv('../data/preprocessed/df_combined_cleaned_sliding_windows.csv',
                                               index=False)


if __name__ == "__main__":
    main()
