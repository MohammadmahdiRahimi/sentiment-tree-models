# Model Architectures

This document provides detailed information about the sentiment classification models implemented in this repository.

## Overview

All models are designed for 5-class sentiment classification on the Stanford Sentiment Treebank dataset, which labels sentences from 0 (very negative) to 4 (very positive).

## Models

### Bag-of-Words (BOW)

**Architecture:**
- Single embedding layer (vocab_size × embedding_dim)
- Sum pooling over word embeddings
- Learned bias vector
- Direct output (no hidden layers)

**Characteristics:**
- Simplest baseline model
- Ignores word order completely
- Fast training and inference
- Limited expressiveness

**Use Case:** Quick baseline for comparison

### Continuous Bag-of-Words (CBOW)

**Architecture:**
- Embedding layer (vocab_size × embedding_dim)
- Sum pooling over embeddings
- Linear projection to output classes
- Softmax for classification

**Characteristics:**
- Extends BOW with learned projection
- Still ignores word order
- Better than BOW due to task-specific projection
- Can use pre-trained embeddings (GloVe, Word2Vec)

**Use Case:** Stronger baseline with semantic word representations

### Deep CBOW

**Architecture:**
- Embedding layer
- Sum pooling
- Two hidden layers with Tanh activation
- Linear output layer

**Layers:**
```
Input → Embedding → Sum → Linear(hidden) → Tanh 
                         → Linear(hidden) → Tanh 
                         → Linear(output)
```

**Characteristics:**
- Non-linear transformations of summed embeddings
- More expressive than shallow CBOW
- Higher risk of overfitting on small datasets
- Benefits from dropout regularization

**Parameters:**
- Embedding dim: 300
- Hidden dim: 100
- Dropout: 0.5 (for pre-trained variant)

**Use Case:** When you need more modeling capacity but can't use sequential models

### LSTM Classifier

**Architecture:**
- Embedding layer
- Custom LSTM cell processing
- Dropout on final hidden state
- Linear output layer

**LSTM Cell:**
- Input gate: i_t = σ(W_ii·x_t + W_hi·h_{t-1})
- Forget gate: f_t = σ(W_if·x_t + W_hf·h_{t-1})
- Cell gate: g_t = tanh(W_ig·x_t + W_hg·h_{t-1})
- Output gate: o_t = σ(W_io·x_t + W_ho·h_{t-1})
- Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
- Hidden state: h_t = o_t ⊙ tanh(c_t)

**Characteristics:**
- Processes sentences sequentially
- Captures word order and context
- Maintains long-term dependencies
- Handles variable-length sequences
- Uses final hidden state for classification

**Parameters:**
- Embedding dim: 300
- Hidden dim: 168
- Dropout: 0.5
- Learning rate: 0.0002

**Use Case:** When word order and sequential dependencies matter

### Tree-LSTM Classifier

**Architecture:**
- Embedding layer
- Leaf transformation for word embeddings
- Tree-LSTM cell for internal nodes
- Dropout on root state
- Linear output layer

**Tree-LSTM Cell:**
- Processes binary tree structures
- Separate forget gates for left and right children
- Input gate: i = σ(W_i·[h_l; h_r])
- Left forget: f_l = σ(W_{fl}·[h_l; h_r])
- Right forget: f_r = σ(W_{fr}·[h_l; h_r])
- Cell gate: g = tanh(W_g·[h_l; h_r])
- Output gate: o = σ(W_o·[h_l; h_r])
- Cell state: c = f_l ⊙ c_l + f_r ⊙ c_r + i ⊙ g
- Hidden state: h = o ⊙ tanh(c)

**Characteristics:**
- Follows syntactic parse tree structure
- Captures compositional semantics
- Models phrase-level sentiment
- Processes hierarchically, not sequentially
- Typically achieves best accuracy

**Parameters:**
- Embedding dim: 300
- Hidden dim: 150
- Dropout: 0.5
- Learning rate: 0.0002
- Batch size: 25

**Use Case:** When syntactic structure is important for understanding sentiment composition

## Performance Comparison

Results on Stanford Sentiment Treebank test set (averaged over 3 random seeds):

| Model | Test Accuracy | Relative Performance | Training Time |
|-------|--------------|---------------------|---------------|
| BOW | 26.1% ± 0.1% | Baseline | Fast (~30s/epoch) |
| CBOW | 37.3% ± 0.5% | Better | Fast (~45s/epoch) |
| Deep CBOW | 37.5% ± 0.3% | Similar to CBOW | Medium (~60s/epoch) |
| PT-DBOW | 40.2% ± 0.5% | Benefits from pre-training | Medium (~60s/epoch) |
| LSTM | 45.3% ± 0.4% | Strong | Slow (~5min/epoch) |
| Tree-LSTM | **46.1% ± 0.2%** | **Best** | Slowest (~8min/epoch) |

**Key Findings:**
- Word order matters: LSTM significantly outperforms BOW variants (+8%)
- Tree structure helps: Tree-LSTM outperforms sequential LSTM (+0.8%)
- Pre-training helps: Models with pre-trained embeddings perform better (+2.7%)
- Sentence length matters: LSTM/Tree-LSTM handle long sentences better than BOW

### Performance by Sentence Length

| Model | Short Sentences | Long Sentences |
|-------|----------------|----------------|
| BOW | 23.9% | 29.0% |
| CBOW | 37.3% | 35.0% |
| PT-DBOW | 43.6% | 37.1% |
| LSTM | 48.4% | 41.4% |
| Tree-LSTM | **48.9%** | **42.6%** |

For complete experimental details, see [RESULTS.md](RESULTS.md).

## Training Tips

### General Guidelines
1. **Start simple**: Begin with BOW/CBOW to establish baselines
2. **Use pre-trained embeddings**: Significant improvement for free
3. **Monitor overfitting**: Deeper models need more regularization
4. **Tune learning rate**: Lower rates (0.0002) work well for LSTM/Tree-LSTM
5. **Batch size**: Larger batches (25) stabilize Tree-LSTM training

### Hyperparameter Recommendations

**BOW/CBOW:**
- Learning rate: 0.0005
- No dropout needed
- Embedding dim: 5 (BOW) or 300 (CBOW)

**Deep CBOW:**
- Learning rate: 0.0005
- Dropout: 0.5
- Hidden dim: 100

**LSTM:**
- Learning rate: 0.0002
- Dropout: 0.5
- Hidden dim: 150-200
- Batch size: 1-25

**Tree-LSTM:**
- Learning rate: 0.0002
- Dropout: 0.5
- Hidden dim: 150
- Batch size: 25 (important for stability)

## Extensions

Possible improvements and extensions:

1. **Attention mechanisms**: Add attention over LSTM outputs
2. **Bidirectional LSTM**: Process sentences in both directions
3. **Multi-task learning**: Train on multiple sentiment datasets
4. **Ensemble methods**: Combine predictions from multiple models
5. **Fine-tuning**: Use modern pre-trained models (BERT, RoBERTa)
6. **Data augmentation**: Generate synthetic training examples

## References

- Socher et al. (2013): "Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank"
- Tai et al. (2015): "Improved Semantic Representations from Tree-Structured Long Short-Term Memory Networks"
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
