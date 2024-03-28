def calculate_accuracy(logits, labels):
    """
    Calculate the accuracy given logits and labels.

    Args:
    - logits (Tensor): Logits output from the model.
    - labels (Tensor): Ground truth labels.

    Returns:
    - accuracy (float): Accuracy as a percentage.
    """
    # Convert logits to predicted labels
    _, predicted_labels = torch.max(logits, 1)
    # Calculate accuracy
    correct_predictions = (predicted_labels == labels).sum().item()
    accuracy = correct_predictions / labels.size(0) * 100
    return accuracy
