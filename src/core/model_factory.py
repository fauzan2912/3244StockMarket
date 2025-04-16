from models.svm import SVMModel
from models.lstm import LSTMModel
from models.attention_lstm import AttentionLSTMModel
from models.rf import RandomForestModel
from models.deep_rnn import DeepRNNModel

def get_model(model_type, **kwargs):
    model_map = {
        "svm": SVMModel,
        "lstm": LSTMModel,
        "attention_lstm": AttentionLSTMModel,
        "rf": RandomForestModel,
        "deep_rnn": DeepRNNModel,
    }
    if model_type not in model_map:
        raise ValueError(f"Unknown model: {model_type}")
    if model_type in ["lstm", "attention_lstm", "deep_rnn"] and "input_shape" not in kwargs:
        raise ValueError(f"{model_type} requires an 'input_shape' argument.")
    return model_map[model_type](**kwargs)
