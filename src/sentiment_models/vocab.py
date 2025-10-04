"""
Vocabulary management for sentiment classification models.

This module provides classes for building and managing vocabularies from text data,
including token frequency counting and bidirectional token-index mappings.
"""

from collections import Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    """
    Counter that remembers insertion order.
    
    Combines the functionality of Counter for counting and OrderedDict
    for maintaining insertion order.
    """

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))


class Vocabulary:
    """
    Vocabulary manager for token-to-index and index-to-token mappings.
    
    Maintains token frequencies and provides methods to build vocabularies
    with special tokens and frequency thresholds.
    
    Attributes:
        freqs: OrderedCounter of token frequencies
        w2i: Dictionary mapping tokens to indices
        i2w: List mapping indices to tokens
    """

    def __init__(self):
        """Initialize an empty vocabulary."""
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        """
        Increment the frequency count for a token.
        
        Args:
            t: Token to count
        """
        self.freqs[t] += 1

    def add_token(self, t):
        """
        Add a token to the vocabulary mappings.
        
        Args:
            t: Token to add
        """
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        """
        Build vocabulary from counted tokens.
        
        Adds special tokens (<unk> and <pad>) first, then adds all tokens
        meeting the minimum frequency threshold in descending frequency order.
        
        Args:
            min_freq: Minimum frequency threshold for including tokens
        """
        self.add_token("<unk>")
        self.add_token("<pad>")
        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.w2i)
