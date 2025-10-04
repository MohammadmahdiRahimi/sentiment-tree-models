# Sentiment Classification with Tree-Structured Models

A PyTorch implementation of various neural architectures for sentence-level sentiment classification on the Stanford Sentiment Treebank (SST). This repository compares multiple approaches from simple bag-of-words models to sophisticated tree-structured LSTMs.

## Overview

This project implements and evaluates several sentiment classification models:

- **Bag-of-Words (BOW)**: Simple baseline using word embeddings
- **Continuous BOW (CBOW)**: Dense embeddings with linear projection
- **Deep CBOW**: Multi-layer non-linear transformations
- **LSTM**: Sequential processing with Long Short-Term Memory
- **Tree-LSTM**: Hierarchical processing following syntactic parse trees

The models are trained and evaluated on the Stanford Sentiment Treebank, which provides fine-grained sentiment labels and parse tree structures for each sentence.

## Features

- Clean, modular PyTorch implementations of multiple sentiment classification architectures
- Support for pre-trained word embeddings (GloVe, Word2Vec)
- Efficient minibatching and data processing utilities
- Comprehensive evaluation metrics and visualization tools
- Extensible vocabulary and embedding management

## Installation

1. Clone this repository:

```bash
git clone https://github.com/MohammadmahdiRahimi/sentiment-tree-models.git
cd sentiment-tree-models
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the Stanford Sentiment Treebank:
   - Visit: https://nlp.stanford.edu/sentiment/
   - Extract to `data/trees/`

4. (Optional) Download pre-trained embeddings:
   - **GloVe**: https://nlp.stanford.edu/projects/glove/
   - **Word2Vec**: https://code.google.com/archive/p/word2vec/
   - Place embedding files in `data/embeddings/`

## Quick Start

```python
from sentiment_models.data import examplereader
from sentiment_models.models import LSTMClassifier
from sentiment_models.vocab import Vocabulary

# Load data
train_data = list(examplereader("data/trees/train.txt"))

# Build vocabulary
vocab = Vocabulary()
vocab.build_from_examples(train_data)

# Initialize model
model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=168,
    output_dim=5,
    vocab=vocab
)

# Train and evaluate
# See scripts/ for complete training examples
```

## Project Structure

```
.
├── src/
│   └── sentiment_models/
│       ├── __init__.py
│       ├── models.py      # Model architectures
│       ├── data.py        # Data loading utilities
│       ├── vocab.py       # Vocabulary management
│       ├── training.py    # Training loops
│       └── utils.py       # Helper functions
├── scripts/
│   ├── quick_run.py       # Quick training example
│   └── extract_pdf.py     # Utility scripts
├── tests/
│   └── test_data.py       # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Model Architectures

### Bag-of-Words (BOW)
Simple baseline that sums word embeddings without considering word order.

### LSTM
Processes sentences sequentially, maintaining hidden states to capture long-range dependencies and context.

### Tree-LSTM
Extends LSTM to hierarchical tree structures, processing sentences according to their syntactic parse trees to capture compositional semantics.

## Results

Comprehensive experiments on the Stanford Sentiment Treebank demonstrate clear performance hierarchies across model architectures.

### Overall Performance

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| BOW | 26.1% ± 0.1% | Simple baseline, ignores word order |
| CBOW | 37.3% ± 0.5% | Learned linear projection |
| Deep CBOW | 37.5% ± 0.3% | Non-linear transformations |
| PT-DBOW | 40.2% ± 0.5% | Pre-trained Word2Vec embeddings |
| LSTM | 45.3% ± 0.4% | Sequential processing |
| **Tree-LSTM** | **46.1% ± 0.2%** | **Hierarchical structure (best)** |

*Results averaged over 3 random seeds (0, 42, 456)*

### Key Insights

📊 **Word Order Impact:** LSTM models outperform CBOW by +8%, demonstrating the critical importance of sequential processing for sentiment classification.

🌳 **Tree Structure Benefit:** Tree-LSTM's hierarchical processing provides a modest but consistent +0.8% improvement over sequential LSTM.

🎯 **Pre-training Helps:** Pre-trained embeddings boost performance by +2.7% (PT-DBOW vs DBOW), showing the value of transfer learning.

📏 **Sentence Length:** LSTM and Tree-LSTM excel on both short (48.9%) and long (42.6%) sentences, while simpler models struggle with longer sequences.

### Detailed Analysis

For comprehensive experimental results including:
- Performance breakdown by sentence length
- Model configuration and hyperparameters
- Training efficiency comparisons
- Robustness analysis across random seeds
- Ablation studies

See **[docs/RESULTS.md](docs/RESULTS.md)**

## Citation

If you use this code, please cite the relevant papers:

**Stanford Sentiment Treebank:**
```bibtex
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  booktitle={EMNLP},
  year={2013}
}
```

**Tree-LSTM:**
```bibtex
@inproceedings{tai2015improved,
  title={Improved semantic representations from tree-structured long short-term memory networks},
  author={Tai, Kai Sheng and Socher, Richard and Manning, Christopher D},
  booktitle={ACL},
  year={2015}
}
```

**LSTM:**
```bibtex
@article{hochreiter1997long,
  title={Long short-term memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural computation},
  year={1997}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
