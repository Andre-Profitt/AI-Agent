import subprocess
import re
import sys
import logging
import os
from typing import List, Dict, Tuple
import pkg_resources
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyManager:
    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = requirements_file
        self.backup_file = f"{requirements_file}.backup"
        self.unpinned_file = f"{requirements_file}.unpinned"
        self.original_requirements = self._read_requirements()
        self.is_huggingface_space = os.environ.get('SPACE_ID') is not None
        
    def _read_requirements(self) -> Dict[str, str]:
        """Read requirements file and return dict of package -> version."""
        requirements = {}
        try:
            with open(self.requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Handle different requirement formats
                        if '==' in line:
                            pkg, version = line.split('==')
                            requirements[pkg] = version
                        elif '>=' in line:
                            pkg, version = line.split('>=')
                            requirements[pkg] = f">={version}"
                        else:
                            requirements[line] = None
        except FileNotFoundError:
            logger.error(f"Requirements file {self.requirements_file} not found")
            return {}
        return requirements

    def _backup_requirements(self):
        """Create a backup of the original requirements file."""
        import shutil
        shutil.copy2(self.requirements_file, self.backup_file)
        logger.info(f"Created backup at {self.backup_file}")

    def _write_requirements(self, requirements: Dict[str, str], filename: str):
        """Write requirements to file."""
        with open(filename, 'w') as f:
            for pkg, version in requirements.items():
                if version:
                    f.write(f"{pkg}=={version}\n")
                else:
                    f.write(f"{pkg}\n")

    def _try_install(self, requirements: Dict[str, str], filename: str) -> bool:
        """Try to install requirements and return success status."""
        self._write_requirements(requirements, filename)
        try:
            # Use --no-deps for Hugging Face Spaces to avoid dependency resolution issues
            cmd = [sys.executable, "-m", "pip", "install", "-r", filename]
            if self.is_huggingface_space:
                cmd.append("--no-deps")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Installation failed: {e.stderr}")
            return False

    def _extract_conflicting_packages(self, error_output: str) -> List[str]:
        """Extract conflicting packages from pip error output."""
        conflicts = []
        # Look for common conflict patterns
        conflict_patterns = [
            r"ERROR: Cannot install ([^ ]+) because these package versions have conflicting dependencies",
            r"ERROR: ResolutionImpossible: for ([^ ]+)",
            r"ERROR: Could not find a version that satisfies the requirement ([^ ]+)"
        ]
        
        for pattern in conflict_patterns:
            matches = re.finditer(pattern, error_output)
            for match in matches:
                conflicts.extend(match.groups())
        
        return list(set(conflicts))

    def install_with_fallback(self) -> Tuple[bool, Dict[str, str]]:
        """
        Try to install requirements with fallback to unpinned versions.
        Returns (success, unpinned_packages)
        """
        self._backup_requirements()
        unpinned_packages = {}
        
        # First try with all pinned versions
        if self._try_install(self.original_requirements, self.requirements_file):
            logger.info("Successfully installed all pinned requirements")
            return True, unpinned_packages

        # If that fails, try unpinning packages one by one
        current_requirements = self.original_requirements.copy()
        
        while True:
            try:
                cmd = [sys.executable, "-m", "pip", "install", "-r", self.requirements_file]
                if self.is_huggingface_space:
                    cmd.append("--no-deps")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    break
                    
                conflicts = self._extract_conflicting_packages(result.stderr)
                if not conflicts:
                    logger.error("Could not identify conflicting packages")
                    break
                    
                # Unpin the first conflicting package
                pkg = conflicts[0]
                if pkg in current_requirements:
                    original_version = current_requirements[pkg]
                    current_requirements[pkg] = None  # Unpin the package
                    unpinned_packages[pkg] = original_version
                    logger.info(f"Unpinned {pkg} from {original_version}")
                    
                    # Try installation again with the unpinned package
                    if self._try_install(current_requirements, self.requirements_file):
                        break
                        
            except Exception as e:
                logger.error(f"Error during installation: {str(e)}")
                break

        # Save the final unpinned requirements
        if unpinned_packages:
            self._write_requirements(current_requirements, self.unpinned_file)
            logger.info(f"Saved unpinned requirements to {self.unpinned_file}")
            
        return len(unpinned_packages) > 0, unpinned_packages

    def restore_original(self):
        """Restore the original requirements file from backup."""
        import shutil
        try:
            shutil.copy2(self.backup_file, self.requirements_file)
            logger.info("Restored original requirements file")
        except FileNotFoundError:
            logger.error("Backup file not found")

    def handle_huggingface_build(self) -> bool:
        """
        Special handling for Hugging Face Space builds.
        Returns True if successful, False otherwise.
        """
        if not self.is_huggingface_space:
            return False

        logger.info("Detected Hugging Face Space build environment")
        
        # Try to install with --no-deps first
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", self.requirements_file, "--no-deps"],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError:
            pass

        # If that fails, try with unpinned versions
        success, unpinned = self.install_with_fallback()
        if success:
            # For Hugging Face Spaces, we want to keep the unpinned version
            if unpinned:
                self._write_requirements(self.original_requirements, self.requirements_file)
            return True
        return False

def main():
    manager = DependencyManager()
    
    # Special handling for Hugging Face Spaces
    if manager.is_huggingface_space:
        if manager.handle_huggingface_build():
            print("✅ Successfully installed dependencies for Hugging Face Space")
            sys.exit(0)
        else:
            print("❌ Failed to install dependencies for Hugging Face Space")
            sys.exit(1)
    
    # Normal handling for local development
    success, unpinned = manager.install_with_fallback()
    
    if success:
        if unpinned:
            print("\nThe following packages were unpinned to resolve conflicts:")
            for pkg, version in unpinned.items():
                print(f"  - {pkg} (was {version})")
            print(f"\nUnpinned requirements saved to {manager.unpinned_file}")
        else:
            print("All requirements installed successfully with pinned versions")
    else:
        print("Failed to resolve all dependency conflicts")
        manager.restore_original()

if __name__ == "__main__":
    main() 