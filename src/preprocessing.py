import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load the dataset from a CSV file and drop the 'Session' column.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded and cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    df.drop(columns=['Session'], inplace=True)
    return df


def preprocess_data(df):
    """
    Preprocess the dataset by encoding categorical variables and scaling numerical variables.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    dict: Dictionary of label encoders for each categorical column.
    """
    X = df.drop(columns=['profit_loss_cum'])
    y = df['profit_loss_cum']
    categorical_columns = ['Participant', 'Group', 'Type']
    numerical_columns = X.columns.difference(categorical_columns)

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns)
        ],
        remainder='passthrough'
    )

    X_preprocessed = preprocessor.fit_transform(X)
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=numerical_columns.tolist() + categorical_columns)

    return X_preprocessed_df, y, label_encoders


def bin_target_variable(y):
    """
    Bin the target variable into discrete classes.

    Args:
    y (pd.Series): The target variable.

    Returns:
    pd.Series: The binned target variable.
    """
    bins = [-float('inf'), 0, 20, 40, 100, float('inf')]
    labels = [0, 1, 2, 3, 4]

    y_binned = pd.cut(y, bins=bins, labels=labels, right=False)
    y_binned = y_binned.astype(int)

    return y_binned


def custom_train_test_split(X, y, participant_column, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets based on unique participants.

    Args:
    X (pd.DataFrame): The feature DataFrame.
    y (pd.Series): The target variable.
    participant_column (str): The column indicating participants.
    test_size (float): The proportion of the dataset to include in the test split.
    val_size (float): The proportion of the training dataset to include in the validation split.
    random_state (int): The random seed.

    Returns:
    tuple: DataFrames and Series for training, validation, and test sets.
    """
    unique_participants = X[participant_column].unique()

    train_val_participants, test_participants = train_test_split(
        unique_participants, test_size=test_size, random_state=random_state, shuffle=True
    )

    train_val_mask = X[participant_column].isin(train_val_participants)
    test_mask = X[participant_column].isin(test_participants)

    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    unique_train_val_participants = X_train_val[participant_column].unique()
    train_participants, val_participants = train_test_split(
        unique_train_val_participants, test_size=val_size, random_state=random_state, shuffle=True
    )

    train_mask = X_train_val[participant_column].isin(train_participants)
    val_mask = X_train_val[participant_column].isin(val_participants)

    X_train = X_train_val[train_mask]
    y_train = y_train_val[train_mask]

    X_val = X_train_val[val_mask]
    y_val = y_train_val[val_mask]

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test, prefix):
    """
    Save the train, validation, and test splits to the specified directories.

    Args:
    X_train, X_val, X_test (pd.DataFrame): The feature DataFrames for the train, val, and test sets.
    y_train, y_val, y_test (pd.Series): The target variables for the train, val, and test sets.
    prefix (str): The prefix for the file paths.
    """
    X_train.to_csv(f'../data/modeling/train/{prefix}_X_train.csv', index=False)
    X_val.to_csv(f'../data/modeling/dev/{prefix}_X_val.csv', index=False)
    X_test.to_csv(f'../data/modeling/test/{prefix}_X_test.csv', index=False)

    y_train.to_csv(f'../data/modeling/train/{prefix}_y_train.csv', index=False)
    y_val.to_csv(f'../data/modeling/dev/{prefix}_y_val.csv', index=False)
    y_test.to_csv(f'../data/modeling/test/{prefix}_y_test.csv', index=False)


def add_gaussian_noise(X, columns, mean=0, std=0.01):
    """
    Add Gaussian noise to the numerical columns.

    Args:
    X (pd.DataFrame): The input DataFrame.
    columns (list): List of numerical columns to add noise to.
    mean (float): Mean of the Gaussian noise.
    std (float): Standard deviation of the Gaussian noise.

    Returns:
    pd.DataFrame: DataFrame with Gaussian noise added to the specified columns.
    """
    noise = np.random.normal(mean, std, X[columns].shape)
    X[columns] += noise
    return X


def main():
    """
    Main function to execute the data loading, preprocessing, binning, splitting, and augmentation.
    """
    # Load data
    file_path = '../data/preprocessed/df_combined_cleaned_sliding_windows.csv'
    df = load_data(file_path)

    # Preprocess data
    X, y, label_encoders = preprocess_data(df)

    # Bin the target variable
    y_binned = bin_target_variable(y)

    # Split the data for regression target
    X_train_df, X_val_df, X_test_df, y_train_df, y_val_df, y_test_df = custom_train_test_split(
        X, y, participant_column='Participant', test_size=0.15, val_size=0.15, random_state=42
    )

    # Split the data for categorical target
    X_train_df_binned, X_val_df_binned, X_test_df_binned, y_train_df_binned, y_val_df_binned, y_test_df_binned = custom_train_test_split(
        X, y_binned, participant_column='Participant', test_size=0.15, val_size=0.15, random_state=42
    )

    # Augment training data for regression
    numerical_columns = X_train_df.columns.difference(['Participant', 'Group', 'Type'])

    # Create augmented training data by duplicating the original training data
    X_train_augmented = pd.concat([X_train_df] * 2, ignore_index=True)
    y_train_augmented = pd.concat([y_train_df] * 2, ignore_index=True)

    # Add Gaussian noise to the duplicated portion of the training data
    X_train_augmented.iloc[len(X_train_df):] = add_gaussian_noise(X_train_augmented.iloc[len(X_train_df):], numerical_columns)

    # Save augmented splits for regression target
    save_splits(X_train_augmented, X_val_df, X_test_df, y_train_augmented, y_val_df, y_test_df, 'regression_augmented')

    # Augment training data for categorical
    X_train_augmented_binned = pd.concat([X_train_df_binned] * 2, ignore_index=True)
    y_train_augmented_binned = pd.concat([y_train_df_binned] * 2, ignore_index=True)

    # Add Gaussian noise to the duplicated portion of the training data
    X_train_augmented_binned.iloc[len(X_train_df):] = add_gaussian_noise(X_train_augmented_binned.iloc[len(X_train_df):], numerical_columns)

    # Save augmented splits for categorical target
    save_splits(X_train_augmented_binned, X_val_df_binned, X_test_df_binned, y_train_augmented_binned, y_val_df_binned, y_test_df_binned, 'categorical_augmented')


# Run the main function
if __name__ == "__main__":
    main()
