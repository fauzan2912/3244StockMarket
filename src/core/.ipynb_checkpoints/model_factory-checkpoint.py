# src/core/model_factory.py

from models.svm import SVMModel
from models.lstm import LstmModel
from models.attention_lstm import AttentionLSTMModel
from models.rf import RandomForestModel
from models.deep_rnn import DeepRNNModel
from models.xgboost import XgboostModel


def get_model(model_type: str, **kwargs):
    """
    Factory function to instantiate different model classes by name.

    Args:
        model_type: One of {'svm', 'lstm', 'attention_lstm', 'rf', 'deep_rnn', 'xgboost'}
        **kwargs: Keyword args forwarded to the model constructor.

    Returns:
        An instance of the requested model class.

    Raises:
        ValueError: If model_type is not recognized.
    """
    model_map = {
        'svm': SVMModel,
        'lstm': LstmModel,
        'attention_lstm': AttentionLSTMModel,
        'rf': RandomForestModel,
        'deep_rnn': DeepRNNModel,
        'xgboost': XgboostModel,
    }

    try:
        ModelClass = model_map[model_type]
    except KeyError:
        valid = ', '.join(model_map.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Valid options are: {valid}.")

    return ModelClass(**kwargs)