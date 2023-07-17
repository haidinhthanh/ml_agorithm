def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.

    Parameters:
        y_true (list or numpy array): The true labels.
        y_pred (list or numpy array): The predicted labels.

    Returns:
        float: The accuracy score.
    """
    # Check if the lengths of y_true and y_pred are the same
    if len(y_true) != len(y_pred):
        raise ValueError("Input lists should have the same length.")

    # Calculate the number of correct predictions
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Calculate the accuracy score
    accuracy = correct_predictions / len(y_true)

    return accuracy

