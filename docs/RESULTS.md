# Experimental Results

This document presents the experimental results and analysis of various sentiment classification models on the Stanford Sentiment Treebank dataset.

## Dataset

**Stanford Sentiment Treebank (SST)**
- Training set: 8,544 examples
- Development set: 1,101 examples  
- Test set: 2,210 examples
- Labels: 5-class (very negative, negative, neutral, positive, very positive)
- Includes binary parse trees and fine-grained sentiment annotations

## Model Performance

All experiments were conducted using three different random seeds (0, 42, 456) to account for variability in model initialization. Results show mean accuracy ± standard deviation.

### Overall Test Accuracy

| Model | Test Accuracy | Std Dev |
|-------|--------------|---------|
| BOW | 26.1% | ±0.1% |
| CBOW | 37.3% | ±0.5% |
| Deep CBOW | 37.5% | ±0.3% |
| PT-DBOW (Pre-trained) | 40.2% | ±0.5% |
| LSTM | 45.3% | ±0.4% |
| Tree-LSTM | **46.1%** | ±0.2% |
| Tree-LSTM (node-level) | 46.2% | ±0.2% |

**Key Finding:** Tree-LSTM achieves the best performance, though the improvement over sequential LSTM is modest (0.8%). This suggests that while syntactic structure helps, sequential processing already captures much of the relevant information.

### Performance by Sentence Length

Models were evaluated on subsets of the test set split by sentence length (threshold at average sentence length):

| Model | Short Sentences | Long Sentences |
|-------|----------------|----------------|
| BOW | 23.9% | 29.0% |
| CBOW | 37.3% | 35.0% |
| Deep CBOW | 37.5% | 35.3% |
| PT-DBOW | 43.6% | 37.1% |
| LSTM | 48.4% | 41.4% |
| Tree-LSTM | **48.9%** | **42.6%** |

**Key Findings:**
- Most models perform better on short sentences due to reduced complexity
- LSTM and Tree-LSTM excel on both short and long sentences
- BOW surprisingly performs *worse* on short sentences - this is because it relies on cumulative word frequencies, which work better when more sentiment-carrying words are present
- The performance gap between simple and sophisticated models is larger on long sentences, highlighting the importance of capturing sequential dependencies

## Analysis

### Impact of Word Order

Models incorporating word order (LSTM: 45.3%, Tree-LSTM: 46.1%) significantly outperform bag-of-words variants (best BOW variant: 40.2%). This demonstrates that:
- Sequential processing captures contextual relationships crucial for sentiment
- Word arrangement influences meaning ("not good" vs "good")
- Order-aware models better handle negation, intensifiers, and compositional phrases

### Impact of Tree Structure

Tree-LSTM (46.1%) vs. Sequential LSTM (45.3%):
- Modest improvement (+0.8%) suggests tree structure provides additional signal
- Hierarchical processing captures phrase-level composition
- Benefit is smaller than expected, possibly because:
  - Sequential LSTMs already capture much compositional information
  - Parse trees may not always align with semantic composition
  - The SST dataset's balanced nature reduces the advantage

### Impact of Pre-trained Embeddings

PT-DBOW (40.2%) vs. DBOW (37.5%):
- Pre-trained embeddings provide +2.7% improvement
- Benefit comes from better semantic representations learned from large corpora
- Reduces the burden of learning embeddings from limited labeled data
- Most significant for models that don't use sequential processing

### Node-Level Supervision

Tree-LSTM with node-level supervision (46.2%) vs. sentence-level only (46.1%):
- Marginal improvement (+0.1%)
- Suggests sentence-level signal is sufficient for root classification
- Added complexity of node-level supervision doesn't translate to better generalization
- May be more valuable for phrase-level sentiment tasks

## Model Configurations

### Training Hyperparameters

**Bag-of-Words Models:**
```
BOW:
- Embedding dim: 5
- Learning rate: 0.0005
- No dropout

CBOW / Deep CBOW:
- Embedding dim: 300
- Hidden dim: 100 (Deep CBOW only)
- Learning rate: 0.0005
- Dropout: 0.5 (PT-DBOW only)
```

**Sequential Models:**
```
LSTM:
- Embedding dim: 300
- Hidden dim: 168
- Learning rate: 0.0002
- Dropout: 0.5
- Batch size: 1

Tree-LSTM:
- Embedding dim: 300
- Hidden dim: 150
- Learning rate: 0.0002
- Dropout: 0.5
- Batch size: 25
```

### Optimization
- Optimizer: Adam
- Loss function: Cross-entropy
- Evaluation metric: Accuracy

## Robustness

Results were consistent across random seeds:
- BOW: σ = 0.001 (very stable)
- Tree-LSTM: σ = 0.002 (very stable)
- PT-DBOW: σ = 0.005 (stable)

Low variance indicates robust model behavior and reliable conclusions.

## Computational Efficiency

Training time per epoch (approximate, on CPU):

| Model | Time per Epoch |
|-------|---------------|
| BOW | ~30 seconds |
| CBOW | ~45 seconds |
| Deep CBOW | ~60 seconds |
| LSTM | ~5 minutes |
| Tree-LSTM | ~8 minutes |

**Trade-off:** More sophisticated models achieve better accuracy but require longer training time.

## Conclusions

1. **Word order matters:** Sequential models (LSTM, Tree-LSTM) vastly outperform bag-of-words approaches
2. **Pre-training helps:** Pre-trained embeddings provide consistent improvements
3. **Tree structure provides modest gains:** Hierarchical processing helps but sequential LSTMs already capture most compositional information
4. **Sentence length affects performance:** All models perform better on shorter sentences except BOW
5. **Node-level supervision has limited benefit:** For sentence-level classification, root supervision is sufficient

## Reproducibility

All results can be reproduced using:
```bash
python scripts/train.py --model [model_name] --seed [0/42/456]
```

Seeds used: 0, 42, 456

For exact hyperparameters, see the model configurations above or refer to `scripts/train.py`.

## Future Improvements

Potential directions for better performance:
- Attention mechanisms over LSTM outputs
- Bidirectional LSTMs
- Pre-trained contextualized embeddings (BERT, RoBERTa)
- Ensemble methods
- Data augmentation
- Multi-task learning with other sentiment datasets
