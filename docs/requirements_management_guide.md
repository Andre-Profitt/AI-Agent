# Requirements Management Guide

## The Problem

Package version conflicts are one of the most frustrating issues in Python development. A single non-existent version can block your entire deployment. Here's how to avoid these issues.

## Key Issues Found

- `llama-index-readers-file==0.5.0` doesn't exist (latest is 0.4.9)
- Version mismatches between related packages (langchain 0.3 with langchain-core 0.1)
- Assuming version numbers without verification

## Solutions

### 1. Use the Verification Script BEFORE Deploying

Save the verification script and run it on your requirements:

```bash
python scripts/verify_requirements.py requirements.txt
```

This will check every package version against PyPI and report issues.

### 2. Use Flexible Version Specifiers

Instead of pinning exact versions, consider:

```txt
# Bad - assumes specific version exists
llama-index-readers-file==0.5.0

# Better - allows compatible versions
llama-index-readers-file>=0.4.0,<0.5.0

# Best for development - latest compatible
llama-index-readers-file~=0.4.9
```

### 3. Keep Related Packages in Sync

LangChain ecosystem packages should use matching major versions:

```txt
langchain==0.3.25
langchain-core==0.3.60  # Same major version (0.3)
langchain-openai==0.3.20
langchain-community==0.3.20
```

### 4. Tools to Check Version Availability

#### Method 1: pip index (requires pip >= 21.2)
```bash
pip index versions langchain-core
```

#### Method 2: PyPI JSON API
```bash
curl -s https://pypi.org/pypi/llama-index-readers-file/json | jq -r '.releases | keys[]' | sort -V | tail -10
```

#### Method 3: pip-compile from pip-tools
```bash
pip install pip-tools
pip-compile requirements.in --generate-hashes
```

### 5. Create a Test Installation Environment

Before deploying:

```bash
# Create fresh environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Test installation
pip install -r requirements.txt

# If successful, you're good to go!
deactivate
rm -rf test_env
```

### 6. Use the Quick Fix Script

For common issues, use the quick fix script:

```bash
python scripts/quick_fix_requirements.py requirements.txt
```

This automatically fixes known problematic versions.

## Preventing Future Issues

### 1. Regular Updates

```bash
# Check for outdated packages
pip list --outdated

# Update specific packages carefully
pip install --upgrade langchain langchain-core
```

### 2. Use Version Ranges

Create a `requirements.in` (source file):

```txt
langchain>=0.3.0,<0.4.0
langchain-core>=0.3.0,<0.4.0
llama-index>=0.12.0,<0.13.0
```

Then compile to requirements.txt:

```bash
pip-compile requirements.in
```

### 3. Document Version Decisions

Add comments explaining why specific versions are chosen:

```txt
# LangChain 0.3.x for Pydantic v2 support
langchain==0.3.25

# Matches llama-index core version
llama-index-readers-file==0.4.9  # Latest as of 2025-06-19
```

### 4. CI/CD Integration

Add version verification to your CI pipeline:

```yaml
# .github/workflows/test.yml
steps:
  - name: Verify requirements
    run: python scripts/verify_requirements.py requirements.txt
  
  - name: Test installation
    run: pip install -r requirements.txt
```

## Emergency Fixes

If you're blocked by version issues:

### Remove version constraints temporarily:
```bash
sed 's/==.*$//' requirements.txt > requirements_no_versions.txt
pip install -r requirements_no_versions.txt
pip freeze > requirements_working.txt
```

### Use the last known working versions:
Keep a `requirements.stable.txt` with your last known working configuration.

### Install from Git for bleeding edge:
```bash
pip install git+https://github.com/langchain-ai/langchain.git
```

## Workflow Integration

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: verify-requirements
        name: Verify requirements
        entry: python scripts/verify_requirements.py
        language: system
        files: requirements.txt
```

### Development Workflow

1. **Before making changes:**
   ```bash
   python scripts/verify_requirements.py requirements.txt
   ```

2. **After updating requirements:**
   ```bash
   python scripts/quick_fix_requirements.py requirements.txt
   python scripts/verify_requirements.py requirements.txt
   pip install -r requirements.txt
   ```

3. **Before committing:**
   ```bash
   python scripts/verify_requirements.py requirements.txt
   ```

## Project-Specific Guidelines

### AI Agent Dependencies

Our project uses several AI/ML libraries that require careful version management:

#### LangChain Ecosystem
- All LangChain packages should use the same major version
- Current stable: 0.3.x series
- Avoid mixing 0.1.x and 0.3.x versions

#### LlamaIndex Ecosystem
- Core packages should match versions
- Readers and integrations should be compatible
- Current stable: 0.12.x series

#### PyTorch and ML Libraries
- Pin specific versions for reproducibility
- Consider CUDA compatibility for GPU builds
- Test with both CPU and GPU environments

### Environment-Specific Requirements

#### Development
```txt
# Development tools
black==23.11.0
flake8==6.1.0
isort==5.12.0
mypy==1.7.1
pytest==7.4.3
```

#### Production
```txt
# Production optimizations
# Remove development tools
# Add production-specific packages
```

#### Hugging Face Spaces
```txt
# Space-specific requirements
# Ensure all packages are available in HF environment
```

## Troubleshooting

### Common Error Messages

#### "No matching distribution found"
- Run verification script to check if version exists
- Use quick fix script for known issues
- Check package name spelling

#### "Conflicting dependencies"
- Use `pip check` to identify conflicts
- Update related packages together
- Consider using `pip-tools` for dependency resolution

#### "ImportError after installation"
- Check if package was installed correctly
- Verify Python version compatibility
- Check for missing system dependencies

### Debugging Commands

```bash
# Check what's installed
pip list

# Check for conflicts
pip check

# Show package details
pip show package_name

# Check available versions
pip index versions package_name

# Install with verbose output
pip install -v package_name==version
```

## Summary

Version management doesn't have to be painful:

✅ **Always verify versions exist before deploying**
✅ **Keep related packages in sync**
✅ **Test in a fresh environment**
✅ **Use tools to automate checking**
✅ **Document your version choices**

The few minutes spent verifying will save hours of debugging deployment failures!

## Tools Reference

### Verification Script
```bash
python scripts/verify_requirements.py [requirements_file]
```

### Quick Fix Script
```bash
python scripts/quick_fix_requirements.py [requirements_file]
```

### Manual Verification
```bash
pip index versions package_name
curl -s https://pypi.org/pypi/package_name/json | jq '.releases | keys[]'
```

### Dependency Resolution
```bash
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
``` 