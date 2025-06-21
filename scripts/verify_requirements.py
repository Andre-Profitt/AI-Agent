#!/usr/bin/env python3
import operator
# TODO: Fix undefined variables: Dict, List, Tuple, available, available_versions, data, e, error, error_info, errors, f, filename, json, line, logging, match, package, package_name, packages, parts, re, requested_version, requirements_text, response, result, results, stable_versions, sys, v, version, version_check, x
# TODO: Fix undefined variables: available, available_versions, data, e, error, error_info, errors, f, filename, line, match, operator, package, package_name, packages, parts, requested_version, requirements_text, response, result, results, stable_versions, v, version, version_check, x

"""
Requirements Version Verifier
This script checks if all packages and versions in a requirements.txt file actually exist on PyPI.
It helps prevent runtime errors from non-existent package versions.
"""

from typing import Tuple
from typing import Dict

import sys
import requests
import re
from typing import List, Tuple, Dict
import json
import logging

logger = logging.getLogger(__name__)

def parse_requirements(self, requirements_text: str) -> List[Tuple[str, str]]:
    """Parse requirements.txt content and extract package names and versions."""
    packages = []
    for line in requirements_text.strip().split('\n'):
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith('#') or line.startswith('=='):
            continue

        # Extract package name and version
        if '==' in line:
            parts = line.split('==')
            if len(parts) == 2:
                package_name = parts[0].strip()
                version = parts[1].strip()
                packages.append((package_name, version))
        elif '>=' in line or '<=' in line or '~=' in line:
            # Handle other version specifiers
            match = re.match(r'^([a-zA-Z0-9-_.]+)\s*([><=~]+)\s*([\d.]+.*)', line)
            if match:
                package_name = match.group(1)
                operator = match.group(2)
                version = match.group(3)
                packages.append((package_name, f"{operator}{version}"))

    return packages

def check_package_version(self, package_name: str, version: str) -> Dict[str, any]:
    """Check if a specific package version exists on PyPI."""
    try:
        # Get package info from PyPI
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=10)

        if response.status_code == 404:
            return {
                "exists": False,
                "error": "Package not found on PyPI",
                "available_versions": []
            }

        if response.status_code != 200:
            return {
                "exists": False,
                "error": f"HTTP {response.status_code}: {response.reason}",
                "available_versions": []
            }

        data = response.json()
        available_versions = list(data.get("releases", {}).keys())

        # Check if the specific version exists
        if version.startswith('=='):
            version_check = version[2:]
        else:
            version_check = version

        if version_check in available_versions:
            return {
                "exists": True,
                "available_versions": sorted(available_versions, reverse=True)[:10]  # Show latest 10
            }
        else:
            return {
                "exists": False,
                "error": f"Version {version_check} not found",
                "available_versions": sorted(available_versions, reverse=True)[:10]
            }

    except requests.exceptions.RequestException as e:
        return {
            "exists": False,
            "error": f"Network error: {str(e)}",
            "available_versions": []
        }
    except json.JSONDecodeError:
        return {
            "exists": False,
            "error": "Invalid response from PyPI",
            "available_versions": []
        }

def verify_requirements(self, requirements_text: str) -> Dict[str, any]:
    """Verify all packages in requirements text."""
    packages = parse_requirements(requirements_text)
    results = {
        "total": len(packages),
        "valid": 0,
        "invalid": 0,
        "errors": []
    }

    logger.info("Checking {} packages...", extra={"len_packages_": len(packages)})
    print("-" * 70)

    for package_name, version in packages:
        result = check_package_version(package_name, version)

        if result["exists"]:
            results["valid"] += 1
            logger.info("✅ {}=={}", extra={"package_name": package_name, "version": version})
        else:
            results["invalid"] += 1
            error_info = {
                "package": package_name,
                "version": version,
                "error": result["error"],
                "available_versions": result["available_versions"][:5]  # Show top 5
            }
            results["errors"].append(error_info)

            logger.info("❌ {}=={}", extra={"package_name": package_name, "version": version})
            logger.info("   Error: {}", extra={"result__error_": result['error']})
            if result["available_versions"]:
                logger.info("   Available versions: {}", extra={"_____join_result__available_versions__": ', '.join(result['available_versions'][:5])})
            logger.info("")

    return results

def suggest_fixes(self, errors: List[Dict]) -> None:
    """Suggest fixes for invalid packages."""
    if not errors:
        return

    print("\n" + "=" * 70)
    logger.info("SUGGESTED FIXES:")
    print("=" * 70)

    for error in errors:
        package = error["package"]
        requested_version = error["version"]
        available = error["available_versions"]

        logger.info("\n{}=={}", extra={"package": package, "requested_version": requested_version})

        if available:
            # Find the closest available version
            if requested_version.replace('==', '') in [v.split('rc')[0].split('b')[0].split('a')[0] for v in available]:
                # Pre-release version might exist
                logger.info("  → Try: {}=={}", extra={"package": package, "available_0": available[0]})
            else:
                # Suggest the latest stable version
                stable_versions = [v for v in available if not any(x in v for x in ['rc', 'a', 'b', 'dev'])]
                if stable_versions:
                    logger.info("  → Try: {}=={}", extra={"package": package, "stable_versions_0": stable_versions[0]})
                else:
                    logger.info("  → Try: {}=={}", extra={"package": package, "available_0": available[0]})
        else:
            logger.info("  → Package might not exist. Check the package name.")

def main():
    """Main function to verify requirements file."""
    if len(sys.argv) > 1:
        # Read from file
        filename = sys.argv[1]
        try:
            with open(filename, 'r') as f:
                requirements_text = f.read()
        except FileNotFoundError:
            logger.info("Error: File '{}' not found", extra={"filename": filename})
            sys.exit(1)
    else:
        # Read from stdin or use the example
        logger.info("Paste your requirements.txt content (press Ctrl+D when done):")
        try:
            requirements_text = sys.stdin.read()
        except KeyboardInterrupt:
            logger.info("\nCancelled")
            sys.exit(0)

    results = verify_requirements(requirements_text)

    print("\n" + "=" * 70)
    logger.info("SUMMARY:")
    print("=" * 70)
    logger.info("Total packages: {}", extra={"results__total_": results['total']})
    logger.info("Valid packages: {} ✅", extra={"results__valid_": results['valid']})
    logger.info("Invalid packages: {} ❌", extra={"results__invalid_": results['invalid']})

    if results["errors"]:
        suggest_fixes(results["errors"])
        sys.exit(1)
    else:
        logger.info("\nAll packages and versions are valid! 🎉")
        sys.exit(0)

if __name__ == "__main__":
    main()
