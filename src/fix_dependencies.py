#!/usr/bin/env python3
from dependency_manager import DependencyManager
import sys

def main():
    # Get requirements file from command line or use default
    requirements_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    
    print(f"Attempting to install dependencies from {requirements_file}")
    print("This will automatically unpin packages if there are conflicts...")
    
    manager = DependencyManager(requirements_file)
    success, unpinned = manager.install_with_fallback()
    
    if success:
        if unpinned:
            print("\n✅ Successfully resolved dependencies!")
            print("\nThe following packages were unpinned to resolve conflicts:")
            for pkg, version in unpinned.items():
                print(f"  - {pkg} (was {version})")
            print(f"\nUnpinned requirements saved to {manager.unpinned_file}")
            print("\nTo restore original pinned versions, run:")
            print(f"  cp {manager.backup_file} {requirements_file}")
        else:
            print("\n✅ All requirements installed successfully with pinned versions")
    else:
        print("\n❌ Failed to resolve all dependency conflicts")
        print("Restoring original requirements file...")
        manager.restore_original()

if __name__ == "__main__":
    main() 