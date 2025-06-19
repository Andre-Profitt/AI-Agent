#!/usr/bin/env python3
"""
Fix final remaining print statements with enhanced patterns
"""

import os
import re
import ast
import logging

# Set up logging with simple format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Files with known remaining prints
TARGET_FILES = {
    "src/core/optimized_chain_of_thought.py": 2,
    "src/core/services/working_memory.py": 3,
    "src/agents/advanced_hybrid_architecture.py": 2,
    "src/utils/tools_introspection.py": 1,
    "src/infrastructure/config/configuration_service.py": 1,
    "src/services/integration_hub_examples.py": 2,
}

class PrintStatementFixer:
    """Enhanced print statement fixer with AST parsing"""
    
    def __init__(self):
        self.total_fixed = 0
        self.files_fixed = 0
    
    def fix_file(self, file_path: str) -> bool:
        """Fix prints in a file using multiple strategies"""
        if not os.path.exists(file_path):
            logger.warning(f"❌ File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Strategy 1: Add imports if missing
            content = self._ensure_imports(content)
            
            # Strategy 2: Use multiple regex patterns
            content = self._apply_regex_fixes(content)
            
            # Strategy 3: AST-based fixing for complex cases
            try:
                content = self._ast_based_fix(content)
            except SyntaxError:
                logger.warning(f"⚠️  Syntax error in {file_path}, using regex only")
            
            # Check if we made changes
            if content != original_content:
                # Count how many prints we fixed
                prints_before = original_content.count('print(')
                prints_after = content.count('print(')
                fixed_count = prints_before - prints_after
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.total_fixed += fixed_count
                self.files_fixed += 1
                
                logger.info(f"✅ Fixed {fixed_count} print statements in {file_path}")
                logger.info(f"   Remaining: {prints_after}")
                return True
            else:
                logger.info(f"ℹ️  No changes made to {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False
    
    def _ensure_imports(self, content: str) -> str:
        """Ensure logging imports are present"""
        lines = content.split('\n')
        
        # Check what's missing
        has_logging_import = any('import logging' in line for line in lines)
        has_logger = any('logger =' in line or 'logger=' in line for line in lines)
        
        # Find insert position (after other imports)
        import_insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith(('import ', 'from ')):
                import_insert_idx = i + 1
        
        # Add logging import if missing
        if not has_logging_import:
            lines.insert(import_insert_idx, "import logging")
            import_insert_idx += 1
        
        # Add logger if missing
        if not has_logger:
            # Skip any empty lines after imports
            while import_insert_idx < len(lines) and not lines[import_insert_idx].strip():
                import_insert_idx += 1
            
            lines.insert(import_insert_idx, "logger = logging.getLogger(__name__)")
            lines.insert(import_insert_idx + 1, "")
        
        return '\n'.join(lines)
    
    def _apply_regex_fixes(self, content: str) -> str:
        """Apply comprehensive regex patterns"""
        
        # Pattern 1: Simple print("string")
        content = re.sub(
            r'\bprint\s*\(\s*"([^"]+)"\s*\)',
            r'logger.info("\1")',
            content
        )
        
        # Pattern 2: Simple print('string')
        content = re.sub(
            r"\bprint\s*\(\s*'([^']+)'\s*\)",
            r'logger.info("\1")',
            content
        )
        
        # Pattern 3: print(f"string {var}")
        def fix_fstring_double(match):
            return self._convert_fstring(match.group(1))
        
        content = re.sub(
            r'\bprint\s*\(\s*f"([^"]+)"\s*\)',
            fix_fstring_double,
            content
        )
        
        # Pattern 4: print(f'string {var}')
        def fix_fstring_single(match):
            return self._convert_fstring(match.group(1))
        
        content = re.sub(
            r"\bprint\s*\(\s*f'([^']+)'\s*\)",
            fix_fstring_single,
            content
        )
        
        # Pattern 5: print(variable)
        content = re.sub(
            r'\bprint\s*\(\s*([a-zA-Z_][\w.]*)\s*\)',
            r'logger.info("Value: %s", \1)',
            content
        )
        
        # Pattern 6: print() empty
        content = re.sub(
            r'\bprint\s*\(\s*\)',
            r'logger.info("")',
            content
        )
        
        # Pattern 7: Multi-line prints
        content = re.sub(
            r'\bprint\s*\(\s*"""([^"]+)"""\s*\)',
            r'logger.info("""\1""")',
            content
        )
        
        # Pattern 8: print("string", var)
        content = re.sub(
            r'\bprint\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
            r'logger.info("\1: %s", \2)',
            content
        )
        
        # Pattern 9: print('string', var)
        content = re.sub(
            r"\bprint\s*\(\s*'([^']+)'\s*,\s*([^)]+)\s*\)",
            r'logger.info("\1: %s", \2)',
            content
        )
        
        # Pattern 10: print with multiple arguments
        content = re.sub(
            r'\bprint\s*\(\s*([^,)]+(?:,\s*[^,)]+)+)\s*\)',
            lambda m: f'logger.info("Values: " + str(({m.group(1)})))',
            content
        )
        
        # Pattern 11: print with string multiplication (e.g., "-" * 50)
        content = re.sub(
            r'\bprint\s*\(\s*"([^"]*)"\s*\*\s*(\d+)\s*\)',
            r'logger.info("\1" * \2)',
            content
        )
        
        # Pattern 12: print with string multiplication single quotes
        content = re.sub(
            r"\bprint\s*\(\s*'([^']*)'\s*\*\s*(\d+)\s*\)",
            r"logger.info('\1' * \2)",
            content
        )
        
        # Pattern 13: print with complex string expressions
        content = re.sub(
            r'\bprint\s*\(\s*"([^"]*)"\s*\+\s*([^)]+)\s*\)',
            r'logger.info("\1" + str(\2))',
            content
        )
        
        # Pattern 14: print with complex string expressions single quotes
        content = re.sub(
            r"\bprint\s*\(\s*'([^']*)'\s*\+\s*([^)]+)\s*\)",
            r"logger.info('\1' + str(\2))",
            content
        )
        
        # Pattern 15: print with method calls (e.g., memory.get_memory_stats())
        content = re.sub(
            r'\bprint\s*\(\s*([a-zA-Z_][\w.]*\([^)]*\))\s*\)',
            r'logger.info("Result: %s", \1)',
            content
        )
        
        # Pattern 16: print with attribute access (e.g., memory.current_state.to_json())
        content = re.sub(
            r'\bprint\s*\(\s*([a-zA-Z_][\w.]*)\s*\)',
            r'logger.info("Value: %s", \1)',
            content
        )
        
        return content
    
    def _convert_fstring(self, fstring_content: str) -> str:
        """Convert f-string to logger format"""
        # Extract variables from f-string
        var_pattern = r'\{([^{}:!]+)(?:[^}]*)?\}'
        variables = re.findall(var_pattern, fstring_content)
        
        # Replace {var} with %s
        message = re.sub(var_pattern, '%s', fstring_content)
        
        if variables:
            # Create argument list
            var_list = ', '.join(variables)
            return f'logger.info("{message}", {var_list})'
        else:
            return f'logger.info("{fstring_content}")'
    
    def _ast_based_fix(self, content: str) -> str:
        """Use AST to find and fix print statements"""
        try:
            tree = ast.parse(content)
            
            class PrintTransformer(ast.NodeTransformer):
                def visit_Call(self, node):
                    # Check if this is a print call
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id == 'print'):
                        
                        # Convert to logger.info
                        logger_attr = ast.Attribute(
                            value=ast.Name(id='logger', ctx=ast.Load()),
                            attr='info',
                            ctx=ast.Load()
                        )
                        
                        # Handle different print arguments
                        if not node.args:
                            # print()
                            node.func = logger_attr
                            node.args = [ast.Constant(value="")]
                        elif len(node.args) == 1:
                            # print(something)
                            arg = node.args[0]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                # print("string")
                                node.func = logger_attr
                            else:
                                # print(variable)
                                node.func = logger_attr
                                node.args = [
                                    ast.Constant(value="Value: %s"),
                                    arg
                                ]
                        else:
                            # print(multiple, args)
                            node.func = logger_attr
                            format_str = "Values: " + " ".join(["%s"] * len(node.args))
                            node.args = [ast.Constant(value=format_str)] + node.args
                    
                    return self.generic_visit(node)
            
            transformer = PrintTransformer()
            new_tree = transformer.visit(tree)
            
            # Convert back to source code
            import astor
            return astor.to_source(new_tree)
            
        except:
            # If AST parsing fails, return original
            return content

def main():
    """Main function to fix all remaining prints"""
    logger.info("🔧 Final Print Statement Fixer")
    logger.info("=" * 60)
    
    fixer = PrintStatementFixer()
    
    # Process target files
    logger.info(f"\n📋 Processing {len(TARGET_FILES)} files with known prints...")
    
    for file_path, expected_prints in TARGET_FILES.items():
        logger.info(f"\n🔍 Processing: {file_path}")
        logger.info(f"   Expected prints: {expected_prints}")
        fixer.fix_file(file_path)
    
    # Also scan for any other files with prints
    logger.info("\n🔍 Scanning for additional files with prints...")
    
    additional_files = []
    for root, dirs, files in os.walk('src'):
        # Skip __pycache__
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if file_path not in TARGET_FILES:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if 'print(' in content:
                            count = content.count('print(')
                            additional_files.append((file_path, count))
                    except:
                        pass
    
    if additional_files:
        logger.info(f"\n📋 Found {len(additional_files)} additional files with prints")
        for file_path, count in additional_files:
            logger.info(f"\n🔍 Processing: {file_path} ({count} prints)")
            fixer.fix_file(file_path)
    
    # Final report
    logger.info("\n" + "=" * 60)
    logger.info("📊 FINAL REPORT")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {fixer.files_fixed}")
    logger.info(f"Total prints fixed: {fixer.total_fixed}")
    
    # Final verification
    logger.info("\n🔍 Final verification...")
    remaining_total = 0
    files_with_prints = []
    
    for root, dirs, files in os.walk('src'):
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    count = content.count('print(')
                    if count > 0:
                        remaining_total += count
                        files_with_prints.append((file_path, count))
                except:
                    pass
    
    if remaining_total == 0:
        logger.info("\n✅ SUCCESS! All print statements have been removed!")
    else:
        logger.info(f"\n⚠️  {remaining_total} print statements still remain:")
        for file_path, count in files_with_prints[:10]:
            logger.info(f"   - {file_path}: {count} prints")
        
        logger.info("\nThese may require manual intervention.")

if __name__ == "__main__":
    main() 