"""
Training and evaluation utilities for sentiment classification models.

This module provides functions for preparing examples, evaluating models,
and running training loops with checkpointing and early stopping.
"""

import time
import torch
from torch import nn
import numpy as np
import random


def prepare_example(example, vocab):
    """
    Prepare a single example for model input.
    
    Converts tokens to indices using the vocabulary and wraps in tensors.
    
    Args:
        example: Example namedtuple with tokens and label
        vocab: Vocabulary object with w2i mapping
        
    Returns:
        Tuple of (input_tensor, target_tensor)
    """
    x = [vocab.w2i.get(t, 0) for t in example.tokens]
    x = torch.LongTensor([x])
    y = torch.LongTensor([example.label])
    return x, y


def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        data: List of examples to evaluate on
        prep_fn: Function to prepare examples
        **kwargs: Additional keyword arguments (unused, for compatibility)
        
    Returns:
        Tuple of (num_correct, num_total, accuracy)
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for example in data:
            x, target = prep_fn(example, model.vocab)
            logits = model(x)
            prediction = logits.argmax(dim=-1)
            correct += (prediction == target).sum().item()
            total += 1
    return correct, total, correct / float(total)


def train_model(model, optimizer, num_iterations=10000,
                print_every=1000, eval_every=1000,
                batch_fn=None,
                prep_fn=prepare_example,
                eval_fn=simple_evaluate,
                batch_size=1, eval_batch_size=None,
                test_data=None,
                train_data=None):
    """
    Train a sentiment classification model.
    
    Runs a training loop with periodic evaluation, checkpointing, and early stopping.
    Saves the best model based on development set accuracy.
    
    Args:
        model: Model to train
        optimizer: PyTorch optimizer
        num_iterations: Total number of training iterations
        print_every: Print training loss every N iterations
        eval_every: Evaluate on dev set every N iterations
        batch_fn: Function to create batches from data
        prep_fn: Function to prepare examples for model input
        eval_fn: Function to evaluate model performance
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size (defaults to batch_size)
        test_data: Test dataset for final evaluation
        train_data: Training dataset
        
    Returns:
        Tuple of (losses, accuracies) tracking training progress
    """
    iter_i = 0
    train_loss = 0.
    start = time.time()
    criterion = nn.CrossEntropyLoss()
    best_eval = 0.
    best_iter = 0
    losses = []
    accuracies = []

    if eval_batch_size is None:
        eval_batch_size = batch_size

    while True:
        for batch in batch_fn(train_data, batch_size=batch_size):
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)
            B = targets.size(0)
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_i += 1
            if iter_i % print_every == 0:
                print(f"Iter {iter_i}: loss={train_loss:.4f}, time={time.time()-start:.2f}s")
                losses.append(train_loss)
                train_loss = 0.
            if iter_i % eval_every == 0:
                _, _, accuracy = eval_fn(model, [], batch_size=eval_batch_size, batch_fn=batch_fn, prep_fn=prep_fn)
                accuracies.append(accuracy)
                print(f"iter {iter_i}: dev acc={accuracy:.4f}")
                if accuracy > best_eval:
                    best_eval = accuracy
                    best_iter = iter_i
                    path = f"{model.__class__.__name__}.pt"
                    ckpt = {"state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_eval": best_eval, "best_iter": best_iter}
                    torch.save(ckpt, path)
            if iter_i == num_iterations:
                print("Done training")
                if test_data is not None:
                    print("Loading best model")
                    path = f"{model.__class__.__name__}.pt"
                    ckpt = torch.load(path)
                    model.load_state_dict(ckpt["state_dict"])
                    _, _, train_acc = eval_fn(model, train_data, batch_size=eval_batch_size, batch_fn=batch_fn, prep_fn=prep_fn)
                    _, _, dev_acc = eval_fn(model, [], batch_size=eval_batch_size, batch_fn=batch_fn, prep_fn=prep_fn)
                    _, _, test_acc = eval_fn(model, test_data, batch_size=eval_batch_size, batch_fn=batch_fn, prep_fn=prep_fn)
                    print(f"best model iter {best_iter}: train acc={train_acc:.4f}, dev acc={dev_acc:.4f}, test acc={test_acc:.4f}")
                return losses, accuracies
