# src/core/model_factory.py

from src.tuning.svm import tune_svm
from src.tuning.logistic import tune_logistic
from src.tuning.rf import tune_random_forest
from src.tuning.lstm import tune_lstm
from src.tuning.attention_lstm import tune_attention_lstm
from src.tuning.deep_rnn import tune_deep_rnn


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
    tuning_map = {
        "svm": tune_svm,
        "logistic": tune_logistic,
        "rf": tune_random_forest,
        "lstm": tune_lstm,
        "attention_lstm": tune_attention_lstm,
        "deep_rnn": tune_deep_rnn,
    }

    try:
        ModelClass = model_map[model_type]
    except KeyError:
        valid = ', '.join(model_map.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Valid options are: {valid}.")

    return ModelClass(**kwargs)