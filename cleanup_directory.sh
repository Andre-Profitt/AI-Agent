#!/bin/bash

echo "🧹 AI Agent Directory Cleanup"
echo "============================"
echo ""

# Create organized structure
echo "📁 Creating organized directory structure..."

# Create _archive directory for old/unused files
mkdir -p _archive/{scripts,examples,docs,reports,old_files}
mkdir -p _active/{saas-platform,core-agent,deployment}

# Move temporary and test files
echo "📦 Moving temporary files..."
mv *.log _archive/old_files/ 2>/dev/null
mv test_*.py _archive/old_files/ 2>/dev/null
mv compare_models.py _archive/old_files/ 2>/dev/null
mv old_agent.py.bak _archive/old_files/ 2>/dev/null

# Move multiple startup scripts to archive (keeping only the working one)
echo "📜 Consolidating startup scripts..."
mv start_frontend*.sh _archive/scripts/ 2>/dev/null
mv start_saas*.sh _archive/scripts/ 2>/dev/null
mv serve_saas.py _archive/scripts/ 2>/dev/null

# Move various guide and report files
echo "📚 Organizing documentation..."
mv AGENT_MIGRATION_GUIDE.md _archive/docs/ 2>/dev/null
mv IMPORT_*.md _archive/docs/ 2>/dev/null
mv VS_CODE_AUTO_IMPORT_GUIDE.md _archive/docs/ 2>/dev/null
mv SECURITY_AUDIT_REPORT.md _archive/docs/ 2>/dev/null
mv CLEANUP_SUMMARY.md _archive/docs/ 2>/dev/null

# Move example files that aren't being used
echo "📝 Archiving unused examples..."
mv examples/demo_*.py _archive/examples/ 2>/dev/null
mv examples/make_agent_super_powerful.py _archive/examples/ 2>/dev/null
mv examples/my_agent.py _archive/examples/ 2>/dev/null
mv examples/super_agent_demo.py _archive/examples/ 2>/dev/null
mv examples/integrated_super_agent.py _archive/examples/ 2>/dev/null
mv examples/multi_api_agent.py _archive/examples/ 2>/dev/null

# Clean up reports directory
echo "📊 Archiving old reports..."
mv reports/audits/*.json _archive/reports/ 2>/dev/null
mv reports/audits/*.txt _archive/reports/ 2>/dev/null

# Move the SaaS platform to active directory
echo "🚀 Organizing active projects..."
mv saas-ui _active/saas-platform/ 2>/dev/null
cp run_simple.sh _active/saas-platform/ 2>/dev/null
cp stop_saas.sh _active/saas-platform/ 2>/dev/null
cp test_server.py _active/saas-platform/ 2>/dev/null
cp README_SAAS.md _active/saas-platform/ 2>/dev/null

# Clean up duplicate and unused requirement files
echo "📋 Consolidating requirements..."
mv requirements_*.txt _archive/old_files/ 2>/dev/null

# Move unused scripts
echo "🔧 Archiving development scripts..."
mv scripts/checks/* _archive/scripts/ 2>/dev/null
mv scripts/fixes/* _archive/scripts/ 2>/dev/null
mv scripts/analysis/* _archive/scripts/ 2>/dev/null

# Remove empty directories
echo "🗑️  Removing empty directories..."
find . -type d -empty -delete 2>/dev/null

# Create a clean README for the organized structure
cat > DIRECTORY_STRUCTURE.md << 'EOF'
# AI Agent Directory Structure

## 📁 _active/
Contains currently active and maintained components:
- **saas-platform/** - The SaaS web interface for AI Agent
- **core-agent/** - Core AI agent implementation
- **deployment/** - Deployment configurations

## 📁 src/
Main source code for the AI Agent system

## 📁 tests/
Test suites for the project

## 📁 _archive/
Contains archived files from development:
- **scripts/** - Old utility scripts
- **examples/** - Example implementations
- **docs/** - Old documentation
- **reports/** - Analysis reports
- **old_files/** - Temporary and backup files

## 🚀 Quick Start
To run the SaaS platform:
```bash
cd _active/saas-platform
./run_simple.sh
```

## 📚 Main Documentation
- README.md - Main project documentation
- _active/saas-platform/README_SAAS.md - SaaS platform guide
EOF

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📊 Summary:"
echo "- Active components moved to _active/"
echo "- Old files archived in _archive/"
echo "- Created DIRECTORY_STRUCTURE.md for reference"
echo ""
echo "💡 Tip: Your SaaS platform is now in _active/saas-platform/"
echo "   Run it with: cd _active/saas-platform && ./run_simple.sh"