"""NumPy activation functions shared by model postprocessors."""
import numpy as np


def softmax(values, axis=-1):
    values = np.asarray(values)
    exponentials = np.exp(values - values.max(axis=axis, keepdims=True))
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def sigmoid(values):
    values = np.asarray(values)
    exponentials = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exponentials), exponentials / (1 + exponentials))
