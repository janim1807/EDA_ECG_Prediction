import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

# plots are commented out



def load_test_data(prefix):
    """
    Load the test data for the given prefix (regression or categorical).

    Args:
    prefix (str): The prefix for the file paths.

    Returns:
    tuple: DataFrame and Series for test features and target variable.
    """
    X_test = pd.read_csv(f'../data/modeling/test/{prefix}_X_test.csv')
    y_test = pd.read_csv(f'../data/modeling/test/{prefix}_y_test.csv').squeeze()
    return X_test, y_test


def load_model(model_path):
    """
    Load the trained model from the specified file path.

    Args:
    model_path (str): The path to the model file.

    Returns:
    model: The loaded model.
    """
    return joblib.load(model_path)


def evaluate_regression_model(model, X_test, y_test):
    """
    Evaluate the regression model on the test set.

    Args:
    model: The trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    """
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print(
        f"Regression Test Metrics: MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R2: {r2_test:.4f}")

    # Create a scatter plot for actual vs. predicted values
   # plt.figure(figsize=(10, 6))
    #plt.scatter(y_test, y_test_pred, alpha=0.5, color='b')
   # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
  #  plt.xlabel('Actual Values')
  ##  plt.ylabel('Predicted Values')
  #  plt.title('Actual vs Predicted Values')
  #  plt.show()


def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate the classification model on the test set.

    Args:
    model: The trained model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    """
    y_test_pred = model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred, average='macro')
    recall_test = recall_score(y_test, y_test_pred, average='macro')
    f1_test = f1_score(y_test, y_test_pred, average='macro')
    # ROC has some problems with the tests, can be added again if ROC values is wanted
    #roc_auc_test = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='macro')

    print(
        f"Classification Test Metrics: Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1: {f1_test:.4f}")

    # Plotting the metrics - Commented out because of the tests because they are hard to test
   # metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    #values = [accuracy_test, precision_test, recall_test, f1_test, roc_auc_test]

   # plt.figure(figsize=(10, 6))
   # plt.bar(metrics, values, color=['blue', 'green', 'red', 'cyan', 'magenta'])
   # plt.xlabel('Metrics')
   # plt.ylabel('Values')
    #plt.title(f'Test Metrics for the {model.__class__.__name__}')
  #  plt.show()

    # Plotting the confusion matrix
    #conf_matrix = confusion_matrix(y_test, y_test_pred)
   # plt.figure(figsize=(8, 6))
   # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
   # plt.xlabel('Predicted')
   # plt.ylabel('Actual')
   # plt.title('Confusion Matrix')
  #  plt.show()

    # Plotting the actual and predicted values as lines
  #  plt.figure(figsize=(14, 7))
   # plt.plot(y_test.reset_index(drop=True), label='Actual Values', linestyle='-', marker='o')
   # plt.plot(pd.Series(y_test_pred), label='Predicted Values', linestyle='-', marker='x')
  #  plt.xlabel('Index')
  #  plt.ylabel('Values')
   # plt.title('Actual vs Predicted Values')
   # plt.legend()
   # plt.show()


def main():
    """
    Main function to load the model and test data, and evaluate the model.
    """
    # Regression model evaluation
    X_test_reg, y_test_reg = load_test_data('regression_augmented')
    X_test_reg = X_test_reg.drop(columns=['Participant'])

    # Load the best regression model - Change here the name to the best model
    regression_model_path = '../models/XGBRegressor_model.pkl'
    regression_model = load_model(regression_model_path)

    # Evaluate the regression model
    evaluate_regression_model(regression_model, X_test_reg, y_test_reg)

    # Classification model evaluation
    X_test_cat, y_test_cat = load_test_data('categorical_augmented')
    X_test_cat = X_test_cat.drop(columns=['Participant'])

    # Load the best classification model - Change here the name to the best model
    classification_model_path = '../models/KNeighbors_model.pkl'
    classification_model = load_model(classification_model_path)

    # Evaluate the classification model
    evaluate_classification_model(classification_model, X_test_cat, y_test_cat)


# Run the main function
if __name__ == "__main__":
    main()
