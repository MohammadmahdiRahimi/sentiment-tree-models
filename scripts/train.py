"""
Train Sentiment Classification Models

This script provides a complete training pipeline for all model architectures.
Supports multiple models, pre-trained embeddings, and comprehensive evaluation.

Usage:
    # Train BOW model
    python scripts/train.py --model bow --epochs 10
    
    # Train LSTM with pre-trained embeddings
    python scripts/train.py --model lstm --pretrained glove --epochs 20
    
    # Train Tree-LSTM
    python scripts/train.py --model treelstm --batch-size 25 --epochs 15

Models: bow, cbow, deepcbow, lstm, treelstm
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from torch import optim

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_models import (
    examplereader,
    Vocabulary,
    BOW,
    CBOW,
    DeepCBOW,
    LSTMClassifier,
    TreeLSTMClassifier,
    get_device,
    set_seed,
    simple_evaluate,
    prepare_example
)


def load_pretrained_embeddings(path, vocab, embedding_dim=300):
    """Load pre-trained embeddings (GloVe or Word2Vec format)."""
    print(f"Loading embeddings from {path}...")
    embeddings = {}
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if len(vector) == embedding_dim:
                embeddings[word] = vector
    
    # Create embedding matrix
    embedding_matrix = np.random.randn(len(vocab), embedding_dim).astype('float32')
    embedding_matrix *= 0.1  # Scale down random initialization
    
    found = 0
    for i, word in enumerate(vocab.i2w):
        if word in embeddings:
            embedding_matrix[i] = embeddings[word]
            found += 1
    
    print(f"Found {found}/{len(vocab)} words ({100*found/len(vocab):.1f}%)")
    return torch.FloatTensor(embedding_matrix)


def create_model(model_name, vocab, args, pretrained_embeddings=None):
    """Create and initialize a model."""
    if model_name == 'bow':
        model = BOW(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            vocab=vocab
        )
    elif model_name == 'cbow':
        model = CBOW(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            num_classes=5,
            vocab=vocab
        )
    elif model_name == 'deepcbow':
        model = DeepCBOW(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_classes=5,
            vocab=vocab
        )
    elif model_name == 'lstm':
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=5,
            vocab=vocab
        )
    elif model_name == 'treelstm':
        model = TreeLSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=5,
            vocab=vocab
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load pre-trained embeddings if provided
    if pretrained_embeddings is not None:
        print("Initializing with pre-trained embeddings...")
        model.embed.weight.data.copy_(pretrained_embeddings)
        if not args.finetune_embeddings:
            model.embed.weight.requires_grad = False
            print("Embeddings frozen (not fine-tuning)")
    
    return model


def train_epoch(model, optimizer, criterion, train_data, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for example in train_data:
        x, target = prepare_example(example, model.vocab)
        x, target = x.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += 1
    
    return total_loss / total, correct / total


def evaluate(model, data, device):
    """Evaluate model on dataset."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in data:
            x, target = prepare_example(example, model.vocab)
            x, target = x.to(device), target.to(device)
            
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += 1
    
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='Train sentiment classification models')
    parser.add_argument('--model', type=str, default='bow',
                       choices=['bow', 'cbow', 'deepcbow', 'lstm', 'treelstm'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=150,
                       help='Hidden dimension for LSTM/DeepCBOW')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pre-trained embeddings')
    parser.add_argument('--finetune-embeddings', action='store_true',
                       help='Fine-tune pre-trained embeddings')
    parser.add_argument('--save-path', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    
    # Load data
    print("\nLoading data...")
    data_dir = os.path.join("data", "trees")
    train_data = list(examplereader(os.path.join(data_dir, "train.txt")))
    dev_data = list(examplereader(os.path.join(data_dir, "dev.txt")))
    test_data = list(examplereader(os.path.join(data_dir, "test.txt")))
    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = Vocabulary()
    for example in train_data:
        for token in example.tokens:
            vocab.count_token(token)
    vocab.build(min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Load pre-trained embeddings if specified
    pretrained_embeddings = None
    if args.pretrained:
        pretrained_embeddings = load_pretrained_embeddings(
            args.pretrained, vocab, args.embedding_dim
        )
    
    # Create model
    print("\nInitializing model...")
    model = create_model(args.model, vocab, args, pretrained_embeddings)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training loop
    print("\nTraining...")
    best_dev_acc = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, optimizer, criterion, train_data, device
        )
        
        # Evaluate
        dev_acc = evaluate(model, dev_data, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Loss={train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, "
              f"Dev Acc={dev_acc:.4f}, "
              f"Time={epoch_time:.2f}s")
        
        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            model_path = os.path.join(args.save_path, f"{args.model}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_acc': dev_acc,
                'args': args
            }, model_path)
            print(f"  → Saved best model (dev acc: {dev_acc:.4f})")
    
    # Final evaluation
    print("\nFinal Evaluation...")
    model_path = os.path.join(args.save_path, f"{args.model}_best.pt")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    train_acc = evaluate(model, train_data, device)
    dev_acc = evaluate(model, dev_data, device)
    test_acc = evaluate(model, test_data, device)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Dev Accuracy:   {dev_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    
    print(f"\n✓ Training complete! Best model saved to {model_path}")


if __name__ == '__main__':
    main()
