# Contributing to eQTL Analysis Pipeline

Thank you for your interest in contributing to the eQTL Analysis Pipeline! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/aa9gj/eQTL_analysis/issues)
2. If not, create a new issue using the Bug Report template
3. Include as much detail as possible:
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment information
   - Error messages/stack traces

### Suggesting Features

1. Check existing issues for similar suggestions
2. Open a new issue using the Feature Request template
3. Describe the use case and proposed solution

### Contributing Code

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
6. Format your code:
   ```bash
   black src tests
   ruff check src tests --fix
   ```
7. Commit your changes with a descriptive message
8. Push to your fork and open a Pull Request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip
- git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/eQTL_analysis.git
cd eQTL_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/eqtl_analysis --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run specific test
pytest tests/test_preprocessing.py::TestPhenotypePreprocessor::test_normalize -v
```

### Code Style

We use the following tools for code quality:

- **black**: Code formatting
- **ruff**: Linting
- **mypy**: Type checking

Run all checks:

```bash
black src tests
ruff check src tests
mypy src
```

## Project Structure

```
eQTL_analysis/
├── src/eqtl_analysis/      # Main package
│   ├── preprocessing/      # Data preprocessing modules
│   ├── analysis/          # eQTL analysis modules
│   ├── utils/             # Utility functions
│   └── cli.py             # Command-line interface
├── tests/                  # Test suite
├── config/                # Example configurations
└── .github/               # GitHub templates and workflows
```

## Pull Request Guidelines

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what changes you made and why
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update documentation if needed
5. **Size**: Keep PRs focused and reasonably sized

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
type: short description

Longer description if needed.
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Formatting changes
- `chore`: Maintenance tasks

## Questions?

Feel free to open an issue for any questions or concerns.

Thank you for contributing!
