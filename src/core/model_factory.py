# src/core/model_factory.py

from models.svm import SVMModel
# from models.logistic import LogisticModel
from models.rf import RandomForestModel

def get_model(model_type, **kwargs):
    model_map = {
        "svm": SVMModel,
        # "logistic": LogisticModel,
        "rf": RandomForestModel,
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model: {model_type}")
    return model_map[model_type](**kwargs)
