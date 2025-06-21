# TODO: Fix undefined variables: cb_files, content, count, f, file, file_path, files, files_with_fstrings, files_with_prints, issue, k, prints, root, self, unprotected, v, verifier
#!/usr/bin/env python3
"""
Final 100% Completion Check - Verify all fixes are complete
"""

import os
import re

class FinalVerifier:
    def __init__(self):
        self.results = {}
        self.issues = []

    def check_print_statements(self):
        """Check for any print statements in src/"""
        print("\n🔍 Checking for print statements...")

        count = 0
        files_with_prints = []

        for root, dirs, files in os.walk('src'):
            if '__pycache__' in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                        prints = content.count('print(')
                        if prints > 0:
                            count += prints
                            files_with_prints.append(file_path)
                    except:
                        pass

        if count == 0:
            print("✅ No print statements found in src/")
            self.results['print_statements'] = 100
        else:
            print(f"❌ Found {count} print statements in {len(files_with_prints)} files")
            self.results['print_statements'] = 0
            self.issues.append(f"Print statements: {count} in {len(files_with_prints)} files")

        return count == 0

    def check_fstring_logging(self):
        """Check for f-string logging"""
        print("\n🔍 Checking for f-string logging...")

        count = 0
        files_with_fstrings = []

        for root, dirs, files in os.walk('src'):
            if '__pycache__' in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                        if re.search(r'logger\.\w+\(f["\"]', content):
                            count += 1
                            files_with_fstrings.append(file_path)
                    except:
                        pass

        if count == 0:
            print("✅ No f-string logging found in src/")
            self.results['fstring_logging'] = 100
        else:
            print(f"❌ Found f-string logging in {count} files")
            self.results['fstring_logging'] = 0
            self.issues.append(f"F-string logging: {count} files")

        return count == 0

    def check_config_protection(self):
        """Check if config access is protected"""
        print("\n🔍 Checking config protection...")

        unprotected = []

        for root, dirs, files in os.walk('src'):
            if '__pycache__' in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()

                        # Check for unprotected is_configured() calls
                        if 'is_configured()' in content and 'is_configured_safe' not in content:
                            unprotected.append(file_path)
                    except:
                        pass

        if not unprotected:
            print("✅ All config access appears protected")
            self.results['config_protection'] = 100
        else:
            print(f"❌ Found {len(unprotected)} files with unprotected config access")
            self.results['config_protection'] = 70
            self.issues.append(f"Unprotected config: {len(unprotected)} files")

        return len(unprotected) == 0

    def check_circuit_breakers(self):
        """Check circuit breaker usage"""
        print("\n🔍 Checking circuit breaker implementation...")

        cb_files = 0
        for root, dirs, files in os.walk('src'):
            if '__pycache__' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        if '@circuit_breaker' in content:
                            cb_files += 1
                    except:
                        pass
        if cb_files > 0:
            print(f"✅ Circuit breaker decorators found in {cb_files} files")
            self.results['circuit_breakers'] = 100
        else:
            print("❌ No circuit breaker decorators found")
            self.results['circuit_breakers'] = 0
            self.issues.append("No circuit breaker decorators found")
        return cb_files > 0

    def summary(self):
        print("\n" + "="*60)
        print("📊 FINAL 100% COMPLETION CHECK")
        print("="*60)
        for k, v in self.results.items():
            print(f"{k}: {v}%")
        if self.issues:
            print("\nIssues:")
            for issue in self.issues:
                print(f" - {issue}")
        else:
            print("\n🎉 All checks passed! 100% completion achieved!")

    def run_all(self):
        self.check_print_statements()
        self.check_fstring_logging()
        self.check_config_protection()
        self.check_circuit_breakers()
        self.summary()

def main():
    print("\n🚀 Running Final 100% Completion Check...")
    verifier = FinalVerifier()
    verifier.run_all()

if __name__ == "__main__":
    main()
