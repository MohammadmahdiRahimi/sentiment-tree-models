# Contributing to Sentiment Tree Models

Thank you for your interest in contributing to this project! This document provides guidelines for contributions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the feature
- Why this feature would be useful
- Possible implementation approaches (if applicable)

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -am 'Add some feature'`)
7. Push to the branch (`git push origin feature/your-feature-name`)
8. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/sentiment-tree-models.git
cd sentiment-tree-models

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings to all functions and classes
- Keep functions focused and concise
- Add type hints where appropriate

### Example:

```python
def prepare_batch(examples: list, vocab: Vocabulary, device: torch.device) -> tuple:
    """
    Prepare a batch of examples for model input.
    
    Args:
        examples: List of Example namedtuples
        vocab: Vocabulary for token-to-index mapping
        device: PyTorch device for tensor placement
        
    Returns:
        Tuple of (input_tensor, target_tensor)
    """
    # Implementation here
    pass
```

## Testing

Before submitting a PR:

```bash
# Run existing tests
python -m pytest tests/

# Add new tests for new features
# Place tests in tests/ directory
```

## Documentation

- Update README.md if you change functionality
- Update docs/USAGE.md for new features
- Update docs/MODELS.md for new architectures
- Add docstrings to all new code
- Include code examples where helpful

## Commit Messages

Write clear, concise commit messages:

```
Add attention mechanism to LSTM model

- Implement multi-head attention layer
- Add attention visualization utilities
- Update documentation with examples
```

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background or identity.

### Our Standards

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community

**Unacceptable behaviors:**
- Harassment, trolling, or derogatory comments
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## Questions?

Feel free to open an issue for:
- Questions about the codebase
- Clarification on contribution guidelines
- Discussion of potential improvements

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.
