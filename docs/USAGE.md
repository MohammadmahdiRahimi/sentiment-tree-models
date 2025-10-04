# Usage Guide

This guide provides examples for using the sentiment classification models.

## Installation

```bash
# Clone the repository
git clone https://github.com/MohammadmahdiRahimi/sentiment-tree-models.git
cd sentiment-tree-models

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Download Stanford Sentiment Treebank

1. Download from: https://nlp.stanford.edu/sentiment/
2. Extract `trainDevTestTrees_PTB.zip`
3. Place files in `data/trees/`:
   - `train.txt`
   - `dev.txt`
   - `test.txt`

### Download Pre-trained Embeddings (Optional)

**GloVe:**
```bash
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d data/embeddings/
```

**Word2Vec:**
```bash
# Download from: https://code.google.com/archive/p/word2vec/
# Place GoogleNews-vectors-negative300.txt in data/embeddings/
```

## Basic Usage

### Loading Data

```python
from sentiment_models import examplereader

# Load training data
train_data = list(examplereader("data/trees/train.txt"))
dev_data = list(examplereader("data/trees/dev.txt"))
test_data = list(examplereader("data/trees/test.txt"))

# Examine an example
example = train_data[0]
print(f"Tokens: {example.tokens}")
print(f"Label: {example.label}")
print(f"Tree: {example.tree}")
```

### Building Vocabulary

```python
from sentiment_models import Vocabulary

# Create and build vocabulary
vocab = Vocabulary()

# Count tokens
for example in train_data:
    for token in example.tokens:
        vocab.count_token(token)

# Build with minimum frequency threshold
vocab.build(min_freq=1)

print(f"Vocabulary size: {len(vocab)}")
```

### Training a Simple Model (CBOW)

```python
from sentiment_models import CBOW, set_seed
import torch

# Set random seed for reproducibility
set_seed(42)

# Initialize model
model = CBOW(
    vocab_size=len(vocab),
    embedding_dim=300,
    num_classes=5,
    vocab=vocab
)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Simple training loop (simplified)
from sentiment_models import train_model, prepare_example

def simple_batch_fn(data, batch_size=1):
    """Simple batching function for sequential processing."""
    for example in data:
        yield example

# Train (this is a simplified version)
losses, accuracies = train_model(
    model=model,
    optimizer=optimizer,
    num_iterations=5000,
    print_every=500,
    eval_every=500,
    batch_fn=simple_batch_fn,
    prep_fn=prepare_example,
    train_data=train_data,
    test_data=test_data
)
```

### Training LSTM Model

```python
from sentiment_models import LSTMClassifier, set_seed
import torch

set_seed(42)

# Initialize LSTM model
model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=168,
    output_dim=5,
    vocab=vocab
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

# Train with the same training function
# (in practice, you'd use more sophisticated batching)
```

### Loading Pre-trained Embeddings

```python
import torch
import numpy as np

def load_embeddings(path, vocab, embedding_dim=300):
    """
    Load pre-trained embeddings and create embedding matrix.
    
    Args:
        path: Path to embeddings file (GloVe or Word2Vec format)
        vocab: Vocabulary object
        embedding_dim: Dimension of embeddings
        
    Returns:
        Embedding matrix as PyTorch tensor
    """
    embeddings = {}
    
    # Load embeddings file
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    
    # Create embedding matrix
    embedding_matrix = np.random.randn(len(vocab), embedding_dim).astype('float32')
    
    found = 0
    for i, word in enumerate(vocab.i2w):
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]
            found += 1
    
    print(f"Found {found}/{len(vocab)} words in pre-trained embeddings")
    
    return torch.FloatTensor(embedding_matrix)

# Load and initialize model with pre-trained embeddings
embedding_matrix = load_embeddings(
    "data/embeddings/glove.840B.300d.txt",
    vocab,
    embedding_dim=300
)

model = CBOW(vocab_size=len(vocab), embedding_dim=300, num_classes=5, vocab=vocab)
model.embed.weight.data.copy_(embedding_matrix)
```

### Evaluation

```python
from sentiment_models import simple_evaluate

# Evaluate on test set
correct, total, accuracy = simple_evaluate(model, test_data)
print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

# Evaluate on specific subsets
def filter_by_length(data, min_len=0, max_len=float('inf')):
    """Filter examples by sentence length."""
    return [ex for ex in data if min_len <= len(ex.tokens) <= max_len]

# Short sentences (< 10 words)
short_test = filter_by_length(test_data, max_len=9)
correct, total, accuracy = simple_evaluate(model, short_test)
print(f"Short sentences: {accuracy:.4f}")

# Long sentences (>= 10 words)
long_test = filter_by_length(test_data, min_len=10)
correct, total, accuracy = simple_evaluate(model, long_test)
print(f"Long sentences: {accuracy:.4f}")
```

### Making Predictions

```python
def predict_sentiment(model, sentence, vocab):
    """
    Predict sentiment for a new sentence.
    
    Args:
        model: Trained model
        sentence: String or list of tokens
        vocab: Vocabulary object
        
    Returns:
        Predicted label (0-4) and class probabilities
    """
    import torch
    
    # Tokenize if needed
    if isinstance(sentence, str):
        tokens = sentence.lower().split()
    else:
        tokens = sentence
    
    # Convert to indices
    indices = [vocab.w2i.get(t, 0) for t in tokens]  # 0 is <unk>
    x = torch.LongTensor([indices])
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).item()
    
    return pred, probs.squeeze().numpy()

# Example usage
sentence = "This movie is absolutely fantastic and amazing"
pred, probs = predict_sentiment(model, sentence, vocab)

labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
print(f"Sentence: {sentence}")
print(f"Prediction: {labels[pred]}")
print(f"Probabilities: {dict(zip(labels, probs))}")
```

## Advanced Usage

### Training Tree-LSTM

```python
from sentiment_models import TreeLSTMClassifier
import torch

# Tree-LSTM requires special batching and preparation
def prepare_treelstm_batch(batch, vocab):
    """Prepare batch for Tree-LSTM model."""
    # This is simplified - real implementation needs proper batching
    if not isinstance(batch, list):
        batch = [batch]
    
    # Get max length for padding
    max_len = max(len(ex.tokens) for ex in batch)
    
    # Prepare inputs
    x_batch = []
    transitions_batch = []
    labels = []
    
    for example in batch:
        # Pad tokens
        indices = [vocab.w2i.get(t, 0) for t in example.tokens]
        indices += [1] * (max_len - len(indices))  # 1 is <pad>
        x_batch.append(indices)
        
        # Get transitions
        transitions_batch.append(example.transitions)
        labels.append(example.label)
    
    x = torch.LongTensor(x_batch)
    y = torch.LongTensor(labels)
    
    # Transpose transitions for batched processing
    max_trans = max(len(t) for t in transitions_batch)
    trans_matrix = []
    for i in range(max_trans):
        trans_matrix.append([t[i] if i < len(t) else 0 for t in transitions_batch])
    
    return (x, trans_matrix), y

# Initialize Tree-LSTM
model = TreeLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=150,
    output_dim=5,
    vocab=vocab
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
```

### Custom Training Loop

```python
import torch
from torch import nn
import time

def custom_train_loop(model, optimizer, train_data, dev_data, 
                     num_epochs=10, batch_size=32):
    """
    Custom training loop with full control.
    """
    criterion = nn.CrossEntropyLoss()
    best_dev_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Training
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Prepare batch
            x, y = prepare_example(batch, vocab)
            
            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for example in dev_data:
                x, target = prepare_example(example, vocab)
                logits = model(x)
                pred = logits.argmax(dim=-1)
                correct += (pred == target).sum().item()
                total += 1
        
        dev_acc = correct / total
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Loss={total_loss:.4f}, "
              f"Dev Acc={dev_acc:.4f}, "
              f"Time={epoch_time:.2f}s")
        
        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  New best model saved!")
```

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce batch size
- Use gradient accumulation
- Reduce model hidden dimensions

**Poor Performance:**
- Try pre-trained embeddings
- Increase model capacity (hidden dim)
- Train for more iterations
- Check learning rate (try lower: 0.0001)

**Slow Training:**
- Use GPU if available
- Increase batch size
- Reduce vocabulary size (higher min_freq)

### Checking Device

```python
from sentiment_models import get_device
import torch

device = get_device()
print(f"Using device: {device}")

# Move model to device
model = model.to(device)

# Move data to device in training loop
x, y = prepare_example(example, vocab)
x = x.to(device)
y = y.to(device)
```

## Examples

See the `scripts/` directory for complete working examples:
- `train.py`: Full-featured training script with command-line arguments

See `tests/test_integration.py` for usage examples of all models.

## Further Reading

- See `docs/MODELS.md` for detailed model architecture information
- Check the original papers for theoretical background
- Explore the notebooks for experimental results and analysis
