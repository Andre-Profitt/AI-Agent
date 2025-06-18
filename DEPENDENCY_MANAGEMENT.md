# Dependency Management Guide

## Overview

This project uses a comprehensive dependency stabilization approach based on expert analysis to ensure secure, reproducible builds and avoid dependency conflicts. The dependencies have been carefully pinned to specific versions that are known to work together and are free from security vulnerabilities.

## Critical Requirements

### Python Version: 3.11

**This project requires Python 3.11.x** - This is not arbitrary but a necessary convergence point:
- **Gradio 5.x requires Python >=3.10**
- **PyTorch 2.0.x supports up to Python 3.11 (not 3.12)**
- **Python 3.11 offers 10-60% performance improvements** over earlier versions

The `.python-version` file enforces this requirement for tools like pyenv.

### Pydantic Version: v2

**This project uses Pydantic v2** - This is mandatory:
- LangChain v0.3.x has fully migrated to Pydantic v2
- Pydantic v1 is no longer supported
- All code must use Pydantic v2 syntax (no `pydantic.v1` imports)

## Dependency Principles

### 1. Exact Version Pinning

All dependencies are pinned to exact versions (`==`) to ensure:
- Deterministic builds across all environments
- No "it works on my machine" issues
- Protection against upstream breaking changes

### 2. Security-First Selection

All pinned versions have been audited against known vulnerabilities:
- **LangChain 0.3.25**: Mitigates OS command injection and SSRF vulnerabilities
- **LlamaIndex 0.12.42**: Patches SQL injection in vector stores
- **Gradio 5.25.2**: Fixes path traversal and ACL bypass issues
- **PyTorch 2.0.1**: Not affected by RCE vulnerability in torch.load()

### 3. Ecosystem Cohesion

Dependencies are grouped into "stable cohorts" - versions released and tested together:
- All `langchain-*` packages aligned to 0.3.x series
- All `llama-index-*` packages aligned to 0.12.42 release
- PyTorch/torchvision/torchaudio aligned per official compatibility matrix

## Workflow

### 1. Installing Dependencies

```bash
# Ensure Python 3.11 is active
python --version  # Should show 3.11.x

# Install from the stabilized requirements
pip install -r requirements.txt
```

### 2. Creating a Lockfile

After successful installation, generate a complete lockfile:

```bash
# Capture all transitive dependencies
pip freeze > requirements.lock

# For reproducible installations
pip install -r requirements.lock --no-deps
```

### 3. Making Dependency Changes

**DO NOT** modify requirements.txt without careful analysis:

1. Check Python version compatibility (must support 3.11)
2. Verify Pydantic v2 compatibility
3. Audit for security vulnerabilities
4. Test with the full dependency graph
5. Update both requirements.txt and regenerate requirements.lock

### 4. Updating Dependencies

For security updates only:
```bash
# Update specific package
pip install --upgrade package==new.version

# Verify compatibility
python -m pytest tests/

# Regenerate lockfile
pip freeze > requirements.lock
```

## Common Issues and Solutions

### NumPy Compatibility

This project uses **NumPy 1.26.4** (last 1.x version) for maximum compatibility. Many libraries have `numpy<2.0` constraints. Do not upgrade to NumPy 2.x without extensive testing.

### PyTorch CUDA Versions

The pinned PyTorch 2.0.1 includes CPU support. For CUDA:
- Consult the [official compatibility matrix](https://pytorch.org/get-started/previous-versions/)
- Maintain version alignment: torch, torchvision, and torchaudio must match

### Hugging Face Spaces

The dependencies are optimized for Hugging Face deployment:
- Fast builds with no dependency resolution needed
- All versions compatible with Spaces' Python environment
- GPU packages included but work on CPU-only spaces

## Security Monitoring

Regular security audits should be performed:

```bash
# Install safety
pip install safety

# Check for vulnerabilities
safety check -r requirements.txt
```

Key packages to monitor:
- LangChain ecosystem (history of prompt injection vulnerabilities)
- Gradio (UI framework with past XSS/SSRF issues)
- PyTorch (deserialization vulnerabilities)
- LlamaIndex (SQL injection in vector stores)

## Long-Term Maintenance

1. **Monthly Security Reviews**: Check for new CVEs in core packages
2. **Quarterly Compatibility Tests**: Verify ecosystem cohesion
3. **Annual Major Updates**: Plan coordinated updates of framework versions
4. **Continuous Monitoring**: Watch for yanked packages on PyPI

## References

- [LangChain v0.3 Migration Guide](https://python.langchain.com/docs/versions/migrating_agentic/)
- [Pydantic v2 Migration](https://docs.pydantic.dev/latest/migration/)
- [PyTorch Compatibility Matrix](https://pytorch.org/get-started/previous-versions/)
- [Python 3.11 Performance](https://docs.python.org/3/whatsnew/3.11.html) 