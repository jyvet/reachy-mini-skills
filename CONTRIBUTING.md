# Contributing to Reachy Mini Skills

Thank you for your interest in contributing to Reachy Mini Skills! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/reachy-mini-skills.git
   cd reachy-mini-skills
   ```
3. Install development dependencies:
   ```bash
   uv sync --all-extras
   ```

## Development Workflow

### Creating a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Before submitting a PR, run:

```bash
# Format code
uv run black src tests

# Lint
uv run ruff check src tests --fix

# Type check
uv run mypy src
```

### Running Tests

Run the test suite with:

```bash
uv run pytest
```

For coverage report:

```bash
uv run pytest --cov=reachy_mini_skills --cov-report=html
```

## Pull Request Process

1. Ensure your code passes all tests and linting checks
2. Update documentation if needed
3. Add tests for new functionality
4. Create a Pull Request with a clear description of changes

### PR Title Convention

Use conventional commit format:

- `feat: Add new feature`
- `fix: Fix bug in component`
- `docs: Update documentation`
- `test: Add tests`
- `refactor: Refactor code`

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Adding New Providers

When adding a new STT, TTS, or LLM provider:

1. Create a new file in the appropriate directory (`src/reachy_mini_skills/stt/`, `tts/`, or `llm/`)
2. Inherit from the base provider class
3. Implement all required abstract methods
4. Add the provider to `__init__.py` exports
5. Update the factory function (e.g., `get_stt_provider`)
6. Add tests for the new provider
7. Update README with provider documentation

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Feel free to open an issue for any questions about contributing.
