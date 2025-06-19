import yaml
from pathlib import Path
from typing import List, defaultdict

class CodebaseAnalyzer:
    """Analyzes codebase for upgrade opportunities"""
    def __init__(self, root_dir: str = ".", patterns_path: str = "analyzer_patterns.yaml"):
        self.root_dir = Path(root_dir)
        self.upgrade_points: List[UpgradePoint] = []
        self.stats = defaultdict(int)
        self.patterns = self._load_patterns(patterns_path)
        self.monitoring_patterns = self.patterns.get("monitoring_patterns", {})
        self.orchestration_patterns = self.patterns.get("orchestration_patterns", {})
        self.testing_patterns = self.patterns.get("testing_patterns", {})

    def _load_patterns(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f) 