"""
Data loading and preprocessing utilities for sentiment classification.

This module provides functions to read and parse the Stanford Sentiment Treebank
dataset, which includes sentence parse trees and sentiment labels.
"""

import re
from collections import namedtuple

# Import only what we need from NLTK to avoid dependency issues
from nltk.tree import Tree

Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])

# Constants for shift-reduce parsing
SHIFT = 0
REDUCE = 1


def filereader(path):
    """
    Read a file line by line and fix escaped backslashes.
    
    Args:
        path: Path to the file to read
        
    Yields:
        Cleaned lines from the file
    """
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """
    Extract the tokens (leaves) from a treestring.
    
    Args:
        s: Treestring representation
        
    Returns:
        List of tokens
    """
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    """
    Compute SHIFT=0 and REDUCE=1 transitions from flattened treestring.
    
    This extracts the sequence of shift-reduce operations needed to construct
    the parse tree in a bottom-up fashion.
    
    Args:
        s: Treestring representation
        
    Returns:
        List of transition integers (0 for SHIFT, 1 for REDUCE)
    """
    s = re.sub(r"\([0-5] ([^)]+)\)", "0", s)
    s = re.sub(r"\)", " )", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\([0-4] ", "", s)
    s = re.sub(r"\)", "1", s)
    return list(map(int, s.split()))


def examplereader(path, lower=False):
    """
    Yield Example objects from a trees file.
    
    Each example contains:
    - tokens: List of word tokens
    - tree: NLTK Tree object representing parse structure
    - label: Sentiment label (0-4, where 0=very negative, 4=very positive)
    - transitions: Sequence of shift-reduce operations
    
    Args:
        path: Path to the trees file
        lower: If True, convert all text to lowercase
        
    Yields:
        Example namedtuples containing parsed data
    """
    for line in filereader(path):
        line_proc = line.lower() if lower else line
        tokens = tokens_from_treestring(line_proc)
        tree = Tree.fromstring(line_proc)
        label = int(line_proc[1])
        trans = transitions_from_treestring(line_proc)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)
