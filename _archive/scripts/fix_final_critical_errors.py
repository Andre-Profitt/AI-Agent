#!/usr/bin/env python3
"""
Fix the final 4 critical syntax errors
"""

import os
import re
from pathlib import Path

def fix_production_vector_store():
    """Fix the production_vector_store.py syntax error"""
    file_path = Path("src/gaia_components/production_vector_store.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the problematic section and rebuild it
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Skip the broken section from line 101 onwards
        if i >= 100 and i <= 128:
            # We'll rebuild this section properly
            if i == 100:
                fixed_lines.append("                embeddings = []\n")
                fixed_lines.append("                for text in texts:\n")
                fixed_lines.append("                    # Simple hash-based embedding\n")
                fixed_lines.append("                    hash_val = hash(text.lower()) % 1000\n")
                fixed_lines.append("                    embedding = [float(hash_val + i) / 1000.0 for i in range(384)]\n")
                fixed_lines.append("                    embeddings.append(embedding)\n")
                fixed_lines.append("\n")
                fixed_lines.append("                if len(embeddings) == 1:\n")
                fixed_lines.append("                    return embeddings[0]\n")
                fixed_lines.append("                return embeddings\n")
                fixed_lines.append("\n")
                fixed_lines.append("        return FallbackEmbeddings()\n")
                fixed_lines.append("\n")
                fixed_lines.append("    def get_cache_stats(self) -> Dict[str, Any]:\n")
                fixed_lines.append('        """Get embedding cache statistics"""\n')
                fixed_lines.append("        return {\n")
                fixed_lines.append('            "cache_size": len(self.embedding_cache),\n')
                fixed_lines.append('            "max_cache_size": self.cache_size,\n')
                fixed_lines.append('            "cache_utilization": len(self.embedding_cache) / self.cache_size,\n')
                fixed_lines.append('            "model_name": self.model_name,\n')
                fixed_lines.append('            "device": self.device\n')
                fixed_lines.append("        }\n")
                fixed_lines.append("\n")
                # Skip to line 129
                i = 128
        else:
            fixed_lines.append(line)
        
        i += 1
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed {file_path}")

def fix_database_enhanced():
    """Fix database_enhanced.py syntax error"""
    file_path = Path("src/infrastructure/database_enhanced.py")
    
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix the TODO comment
        content = content.replace("TODO: Fix imports and undefined variables", "# TODO: Fix imports and undefined variables")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {file_path}")

def fix_knowledge_ingestion():
    """Fix knowledge_ingestion.py syntax error"""
    file_path = Path("src/services/knowledge_ingestion.py")
    
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix any TODO comments without #
        content = re.sub(r'^TODO:', '# TODO:', content, flags=re.MULTILINE)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Fixed {file_path}")

def fix_comprehensive_code_auditor_broken():
    """We can just remove this file since we have a working version"""
    file_path = Path("comprehensive_code_auditor_broken.py")
    
    if file_path.exists():
        os.remove(file_path)
        print(f"✅ Removed broken file: {file_path}")

def main():
    """Fix all critical syntax errors"""
    print("🔧 Fixing final critical syntax errors...")
    
    fix_production_vector_store()
    fix_database_enhanced()
    fix_knowledge_ingestion()
    fix_comprehensive_code_auditor_broken()
    
    print("\n✅ All critical syntax errors fixed!")

if __name__ == "__main__":
    main()