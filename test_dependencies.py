#!/usr/bin/env python3
"""
Dependency Test Script
Tests that all core dependencies can be imported and are the correct versions.
"""

import sys
import importlib
from typing import List, Tuple
import warnings

# Suppress specific warnings during import testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def test_import(module_name: str, expected_version: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported and optionally check its version."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        
        if expected_version and version != expected_version and version != 'Unknown':
            return False, f"Version mismatch: expected {expected_version}, got {version}"
        
        return True, f"v{version}"
    except ImportError as e:
        return False, f"Import failed: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("=" * 60)
    print("Dependency Test Report")
    print(f"Python Version: {sys.version}")
    print("=" * 60)
    
    # Core dependencies to test
    core_deps = [
        ("pydantic", "2.8.2"),
        ("langchain", "0.3.25"),
        ("langgraph", "0.4.8"),
        ("langchain_core", "0.3.65"),
        ("langchain_community", "0.3.25"),
        ("langchain_openai", "0.3.23"),
        ("llama_index", "0.12.42"),
        ("llama_index.core", "0.12.42"),
        ("gradio", "5.25.2"),
        ("torch", "2.0.1"),
        ("numpy", "1.26.4"),
        ("pandas", "2.2.3"),
        ("openai", None),
        ("groq", None),
        ("supabase", None),
    ]
    
    # Track results
    passed = 0
    failed = 0
    warnings_list = []
    
    print("\nTesting Core Dependencies:")
    print("-" * 60)
    
    for module_name, expected_version in core_deps:
        success, result = test_import(module_name, expected_version)
        
        if success:
            if "mismatch" in result:
                status = "⚠️  WARNING"
                warnings_list.append((module_name, result))
            else:
                status = "✅ PASS"
                passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{status:12} {module_name:30} {result}")
    
    # Test Pydantic v2 specific features
    print("\n\nTesting Pydantic v2 Features:")
    print("-" * 60)
    
    try:
        from pydantic import BaseModel, field_validator
        print("✅ PASS     Pydantic v2 imports (BaseModel, field_validator)")
        passed += 1
    except ImportError as e:
        print(f"❌ FAIL     Pydantic v2 imports: {e}")
        failed += 1
    
    # Test for legacy Pydantic v1 imports (should fail)
    try:
        from pydantic.v1 import BaseModel as V1BaseModel
        print("⚠️  WARNING  pydantic.v1 is available (legacy code may exist)")
        warnings_list.append(("pydantic.v1", "Legacy v1 imports available"))
    except ImportError:
        print("✅ PASS     No pydantic.v1 imports (good!)")
        passed += 1
    
    # Test critical integrations
    print("\n\nTesting Critical Integrations:")
    print("-" * 60)
    
    integrations = [
        ("langchain_groq", None),
        ("langchain_tavily", None),
        ("llama_index.embeddings.openai", None),
        ("llama_index.vector_stores.supabase", None),
    ]
    
    for module_name, _ in integrations:
        success, result = test_import(module_name, None)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:12} {module_name:40} {result}")
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Warnings: {len(warnings_list)}")
    
    if warnings_list:
        print("\nWarnings:")
        for module, warning in warnings_list:
            print(f"  - {module}: {warning}")
    
    print("=" * 60)
    
    # Exit code based on failures
    sys.exit(1 if failed > 0 else 0)

if __name__ == "__main__":
    main() 