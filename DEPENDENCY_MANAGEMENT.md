# Dependency Management Guide

## Overview

This project uses a best-practice approach to dependency management to ensure fast, reproducible builds and avoid dependency conflicts.

## Quick Fix Applied

Fixed the immediate conflict where `httpx==0.28.1` requires `httpcore==1.*` but `httpcore==0.17.3` was pinned:
- Updated `httpcore==0.17.3` → `httpcore==1.0.5` in requirements.txt

## Python Version Compatibility

**Important**: This project requires **Python 3.8-3.12**. Python 3.13 is not yet supported by many dependencies, particularly:
- PyTorch (torch, torchvision)
- Some LlamaIndex packages
- Various ML/AI libraries

### For Hugging Face Spaces
The default Python version should work. If you encounter issues, ensure the Space is using Python 3.11 or 3.12.

## Best Practice Approach

### 1. Two-File System

- **`requirements.in`**: Contains only direct dependencies (packages we explicitly use)
- **`requirements.txt`**: Auto-generated file with all dependencies pinned to exact versions

### 2. Workflow

1. **Install pip-tools locally**:
   ```bash
   pip install pip-tools
   ```

2. **Generate locked requirements**:
   ```bash
   pip-compile requirements.in
   ```

3. **Update dependencies**:
   ```bash
   pip-compile --upgrade requirements.in
   ```

4. **Install in production**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Benefits

- **Fast Builds**: No dependency resolution needed during Docker builds
- **Reproducible**: Same exact versions installed everywhere
- **No Conflicts**: pip-compile resolves all dependencies upfront
- **Easy Updates**: Simply run pip-compile --upgrade and test

### 4. Development Workflow

1. Add new dependencies to `requirements.in`
2. Run `pip-compile requirements.in`
3. Test the application
4. Commit both `requirements.in` and `requirements.txt`

### 5. CI/CD

Your Dockerfile should use:
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

This ensures fast, deterministic builds without dependency resolution overhead. 