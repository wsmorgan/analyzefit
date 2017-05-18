"""Tools for data minipulation."""
import numpy as np

def residual(y,pred):
    """Finds the residual of the actual vs the predicted values.

    Args:
        y (numpy.array): An array containing the correct values of the model.
        pred (numpy.array): An array containing the predicted values of the model.

    Returns:
        residaul (numpy.array): The residual of the data (y-pred).

    Raises:
        ValueError: Raises a value error if y and pred don't have the same number of elements.
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)
        
    if not isinstance(pred, np.ndarray):
        y = np.array(pred)

    if len(y)!=len(pred):
        raise ValueError("Both y and pred must have the same number of elements.")

    return y-pred

def std_residuals(y,pred):
    """Finds the residual of the actual vs the predicted values.

    Args:
        y (numpy.array): An array containing the correct values of the model.
        pred (numpy.array): An array containing the predicted values of the model.

    Returns:
        standardized_residaul (numpy.array): The standardazied residual of the data (y-pred).
    """

    res = residual(y,pred)
    std = np.std(res)

    return res/std
    
    
