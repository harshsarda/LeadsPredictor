import numpy as np


def get_predictions(models, data):
    """
    models: List of models
    data: np.array
    """
    return np.expm1(np.mean([i.predict(data) for i in models], axis=0))
