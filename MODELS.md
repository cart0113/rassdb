# Supported Embedding Models in RASSDB

RASSDB supports exactly 4 code embedding models, each optimized for different use cases:

## 1. CodeRankEmbed (Recommended Default)
**Model ID:** `nomic-ai/CodeRankEmbed`

### Why It's Recommended
- **Best for AI coding agents** that need to understand existing code and respond to natural language queries
- Specifically designed for code retrieval from natural language queries
- Outperforms OpenAI-v3-large and CodeSage-large by 13.80% and 16.81% on code retrieval
- Supports 8192 token context length, allowing for larger code chunks
- Simple integration with no complex metadata preparation needed

### Usage Strategy
- **Code chunks:** Index raw code without modification
- **Queries:** Prefix with "Represent this query for searching relevant code: "
- **Ideal chunk size:** 100-400 lines or 2,000-10,000 characters

### Example
```python
# For queries:
query = "Represent this query for searching relevant code: add input validation to user registration"

# For code (no modification):
def register_user(username, email, password):
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return user
```

## 2. Nomic Embed Code
**Model ID:** `nomic-ai/nomic-embed-code`

### Best For
- Code-to-code similarity search
- Finding similar implementations or patterns
- When you don't need natural language query support

### Usage Strategy
- **Raw code only** - no metadata needed
- Model is trained on GitHub-style code and infers context automatically
- **Ideal chunk size:** 200-500 lines or 2,000-5,000 characters

### Note
This model is not included in the download script due to size constraints but is fully supported by RASSDB.

## 3. CodeBERT
**Model ID:** `microsoft/codebert-base`

### Best For
- Natural language to code search when documentation is important
- Projects with rich docstrings and comments
- Smaller codebases where chunk size limits aren't an issue

### Usage Strategy
- Include **comprehensive docstrings** with implementation
- No artificial metadata - just natural code + documentation
- **Ideal chunk size:** 50-150 lines or 1,500-3,000 characters

### Example
```python
def calculate_fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    This implementation uses an iterative approach with O(1) space complexity
    instead of the recursive approach to avoid stack overflow for large values.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
    """
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

## 4. Qodo-Embed-1.5B
**Model ID:** `Qodo/Qodo-Embed-1-1.5B`

### Best For
- Structured code RAG systems
- When you need rich metadata for filtering
- Large-scale codebases with complex organization

### Usage Strategy
- Add **explicit metadata headers** as comments
- Include file path, class, method, and language information
- **Ideal chunk size:** 100-300 lines or 3,000-8,000 characters

### Example
```python
# File: src/utils/algorithms.py
# Class: MathUtils
# Method: calculate_fibonacci
# Language: Python
def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number iteratively."""
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

## Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| AI coding assistant | CodeRankEmbed | Optimized for NL→code queries, large context |
| Code similarity search | Nomic Embed Code | Best code-to-code similarity |
| Documentation-heavy projects | CodeBERT | Excels with rich docstrings |
| Enterprise RAG systems | Qodo-Embed-1.5B | Structured metadata support |

## Downloading Models

Use the provided script to download the models (except Nomic Embed Code due to size):

```bash
python download_models.py
```

This will download:
- CodeBERT (~420MB)
- Qodo-Embed-1.5B (~6.2GB)
- CodeRankEmbed (~550MB)

## Using a Specific Model

You can specify which model to use during indexing:

```bash
# Use the default (CodeRankEmbed)
rassdb index /path/to/code

# Use a specific model
rassdb index --model microsoft/codebert-base /path/to/code
rassdb index --model Qodo/Qodo-Embed-1-1.5B /path/to/code
rassdb index --model nomic-ai/nomic-embed-code /path/to/code
```

## Error Handling

If you try to use an unsupported model, RASSDB will show an error with the list of supported models. Only the 4 models listed above are supported - no other embedding models will work with RASSDB.