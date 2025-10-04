"""
Sentiment classification model architectures.

This module implements various neural network architectures for sentiment 
classification, ranging from simple bag-of-words to sophisticated tree-structured LSTMs.
"""

import torch
from torch import nn
import math


class BOW(nn.Module):
    """
    Bag-of-Words sentiment classifier.
    
    Sums word embeddings and applies a linear transformation to produce logits.
    This model ignores word order and serves as a simple baseline.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        vocab: Vocabulary object for word-to-index mapping
    """
    def __init__(self, vocab_size, embedding_dim, vocab):
        super(BOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)

    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of word indices, shape (batch_size, seq_length)
            
        Returns:
            Logits for each class, shape (batch_size, embedding_dim)
        """
        embeds = self.embed(inputs)
        logits = embeds.sum(1) + self.bias
        return logits


class CBOW(nn.Module):
    """
    Continuous Bag-of-Words sentiment classifier.
    
    Extends BOW with a linear projection layer for classification.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        num_classes: Number of output classes
        vocab: Vocabulary object for word-to-index mapping
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, vocab):
        super(CBOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of word indices, shape (batch_size, seq_length)
            
        Returns:
            Logits for each class, shape (batch_size, num_classes)
        """
        embeds = self.embed(inputs)
        embeds_sum = embeds.sum(1)
        logits = self.linear(embeds_sum)
        return logits


class DeepCBOW(nn.Module):
    """
    Deep Continuous Bag-of-Words sentiment classifier.
    
    Extends CBOW with multiple non-linear hidden layers for learning
    more complex representations.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of hidden layers
        num_classes: Number of output classes
        vocab: Vocabulary object for word-to-index mapping
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, vocab):
        super(DeepCBOW, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.sequence = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, inputs):
        """
        Forward pass.
        
        Args:
            inputs: Tensor of word indices, shape (batch_size, seq_length)
            
        Returns:
            Logits for each class, shape (batch_size, num_classes)
        """
        embeds = self.embed(inputs)
        embeds_sum = embeds.sum(1)
        logits = self.sequence(embeds_sum)
        return logits


class MyLSTMCell(nn.Module):
    """
    Custom LSTM cell implementation.
    
    Implements the standard LSTM equations with input, forget, cell, and output gates.
    
    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        bias: Whether to use bias parameters
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(MyLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ii = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_if = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_ig = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_io = nn.Linear(input_size, hidden_size, bias=bias)
        self.W_hi = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_hf = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_hg = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.W_ho = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_, hx, mask=None):
        """
        Forward pass through LSTM cell.
        
        Args:
            input_: Input tensor at current timestep
            hx: Tuple of (hidden_state, cell_state) from previous timestep
            mask: Optional mask for selective updates
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        prev_h, prev_c = hx
        i = self.W_ii(input_) + self.W_hi(prev_h)
        f = self.W_if(input_) + self.W_hf(prev_h)
        g = self.W_ig(input_) + self.W_hg(prev_h)
        o = self.W_io(input_) + self.W_ho(prev_h)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * prev_c + i * g
        h = o * torch.tanh(c)
        return h, c


class LSTMClassifier(nn.Module):
    """
    LSTM-based sentiment classifier.
    
    Processes sentences sequentially using LSTM cells, then classifies based on
    the final hidden state (or last non-padding position in batches).
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        output_dim: Number of output classes
        vocab: Vocabulary object for word-to-index mapping
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through LSTM sequence.
        
        Args:
            x: Tensor of word indices, shape (batch_size, seq_length)
            
        Returns:
            Logits for each class, shape (batch_size, output_dim)
        """
        B = x.size(0)
        T = x.size(1)
        input_ = self.embed(x)
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)
        outputs = []
        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)
        if B == 1:
            final = hx
        else:
            outputs = torch.stack(outputs, dim=0)
            outputs = outputs.transpose(0, 1).contiguous()
            pad_positions = (x == 1).unsqueeze(-1)
            outputs = outputs.contiguous()
            outputs = outputs.masked_fill_(pad_positions, 0.)
            mask = (x != 1)
            lengths = mask.sum(dim=1)
            indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]
        logits = self.output_layer(final)
        return logits


class TreeLSTMCell(nn.Module):
    """
    Tree-LSTM cell for processing binary tree structures.
    
    Combines hidden and cell states from left and right children using
    separate forget gates for each child.
    
    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden state
        bias: Whether to use bias parameters
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_layer = nn.Linear(2 * hidden_size, 5 * hidden_size)
        self.dropout_layer = nn.Dropout(p=0.25)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with uniform distribution."""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, hx_l, hx_r, mask=None):
        """
        Forward pass combining left and right child states.
        
        Args:
            hx_l: Tuple of (hidden_state, cell_state) for left child
            hx_r: Tuple of (hidden_state, cell_state) for right child
            mask: Optional mask for selective updates
            
        Returns:
            Tuple of (parent_hidden_state, parent_cell_state)
        """
        prev_h_l, prev_c_l = hx_l
        prev_h_r, prev_c_r = hx_r
        children = torch.cat([prev_h_l, prev_h_r], dim=1)
        proj = self.reduce_layer(children)
        i, f_l, f_r, g, o = torch.chunk(proj, 5, dim=-1)
        i = torch.sigmoid(i)
        f_l = torch.sigmoid(f_l)
        f_r = torch.sigmoid(f_r)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f_l * prev_c_l + f_r * prev_c_r + i * g
        h = o * torch.tanh(c)
        return h, c


class TreeLSTM(nn.Module):
    """
    Tree-LSTM module for processing sequences according to parse trees.
    
    Processes input embeddings following a sequence of shift-reduce transitions
    that define the binary parse tree structure.
    
    Args:
        input_size: Dimension of input embeddings
        hidden_size: Dimension of hidden states
        bias: Whether to use bias parameters
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(TreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce = TreeLSTMCell(input_size, hidden_size)
        self.proj_x = nn.Linear(input_size, hidden_size)
        self.proj_x_gate = nn.Linear(input_size, hidden_size)
        self.buffers_dropout = nn.Dropout(p=0.5)

    def forward(self, x, transitions):
        """
        Forward pass through tree structure.
        
        Args:
            x: Input embeddings, shape (batch_size, seq_length, embedding_dim)
            transitions: List of transition sequences defining tree structure
            
        Returns:
            Root node hidden states, shape (batch_size, hidden_size)
        """
        B = x.size(0)
        T = x.size(1)
        buffers_c = self.proj_x(x)
        buffers_h = buffers_c.tanh()
        buffers_h_gate = self.proj_x_gate(x).sigmoid()
        buffers_h = buffers_h_gate * buffers_h
        buffers = torch.cat([buffers_h, buffers_c], dim=-1)
        D = buffers.size(-1) // 2
        buffers = buffers.split(1, dim=0)
        buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]
        stacks = [[] for _ in buffers]
        for t_batch in transitions:
            child_l = []
            child_r = []
            for transition, buffer, stack in zip(t_batch, buffers, stacks):
                if transition == 0:  # SHIFT
                    stack.append(buffer.pop())
                elif transition == 1:  # REDUCE
                    child_r.append(stack.pop())
                    child_l.append(stack.pop())
            if child_l:
                reduced = iter(torch.split(torch.cat([s for s in child_l + child_r], 0).chunk(2, 1)[0], 1, 0))
                for transition, stack in zip(t_batch, stacks):
                    if transition == 1:
                        stack.append(next(reduced))
        final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
        final = torch.cat(final, dim=0)
        return final


class TreeLSTMClassifier(nn.Module):
    """
    Tree-LSTM sentiment classifier.
    
    Processes sentences according to their syntactic parse trees, capturing
    compositional semantics through hierarchical structure.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of Tree-LSTM hidden states
        output_dim: Number of output classes
        vocab: Vocabulary object for word-to-index mapping
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(TreeLSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.treelstm = TreeLSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self, x):
        """
        Forward pass through Tree-LSTM.
        
        Args:
            x: Tuple of (word_indices, transitions)
                word_indices: shape (batch_size, seq_length)
                transitions: List of transition sequences
            
        Returns:
            Logits for each class, shape (batch_size, output_dim)
        """
        x, transitions = x
        emb = self.embed(x)
        root_states = self.treelstm(emb, transitions)
        logits = self.output_layer(root_states)
        return logits
