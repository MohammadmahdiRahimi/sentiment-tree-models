"""
Sentiment Tree Models

A PyTorch package for sentiment classification using various neural architectures,
from simple bag-of-words to sophisticated tree-structured LSTMs.

This package provides:
- Multiple model architectures (BOW, CBOW, DeepCBOW, LSTM, TreeLSTM)
- Data loading utilities for Stanford Sentiment Treebank
- Vocabulary management
- Training and evaluation utilities
- Support for pre-trained word embeddings
"""

# Lazy imports to avoid dependency issues on import
def __getattr__(name):
    if name in ["filereader", "examplereader", "tokens_from_treestring", "Example"]:
        from .data import filereader, examplereader, tokens_from_treestring, Example
        return locals()[name]
    elif name == "Vocabulary":
        from .vocab import Vocabulary
        return Vocabulary
    elif name in ["BOW", "CBOW", "DeepCBOW", "LSTMClassifier", "TreeLSTMClassifier"]:
        from .models import BOW, CBOW, DeepCBOW, LSTMClassifier, TreeLSTMClassifier
        return locals()[name]
    elif name in ["train_model", "simple_evaluate", "prepare_example"]:
        from .training import train_model, simple_evaluate, prepare_example
        return locals()[name]
    elif name in ["get_device", "set_seed"]:
        from .utils import get_device, set_seed
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"

__all__ = [
    "filereader",
    "examplereader",
    "tokens_from_treestring",
    "Example",
    "Vocabulary",
    "BOW",
    "CBOW",
    "DeepCBOW",
    "LSTMClassifier",
    "TreeLSTMClassifier",
    "train_model",
    "simple_evaluate",
    "prepare_example",
    "get_device",
    "set_seed",
]
