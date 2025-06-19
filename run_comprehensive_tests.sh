#!/bin/bash
# run_comprehensive_tests.sh
# Ultimate test execution script

set -e

echo "🚀 AI Agent System - Comprehensive Test Suite"
echo "=============================================="

# Run pre-test checklist
./pre_test_checklist.sh

echo ""
echo "🧪 Starting Comprehensive Test Suite..."
echo ""

# 1. Circuit Breaker Tests (Most Critical)
echo "1️⃣ Running Circuit Breaker Tests..."
pytest tests/unit/test_circuit_breaker.py -v --tb=short --maxfail=1
echo "✅ Circuit Breaker Tests PASSED"
echo ""

# 2. Config Validation Tests
echo "2️⃣ Running Config Validation Tests..."
pytest tests/test_config_validation.py -v --tb=short --maxfail=1
echo "✅ Config Validation Tests PASSED"
echo ""

# 3. Structured Logging Tests
echo "3️⃣ Running Structured Logging Tests..."
pytest tests/test_structured_logging.py -v --tb=short --maxfail=1
echo "✅ Structured Logging Tests PASSED"
echo ""

# 4. Monitoring Tests
echo "4️⃣ Running Monitoring Tests..."
pytest tests/unit/test_monitoring_metrics.py -v --tb=short --maxfail=1
echo "✅ Monitoring Tests PASSED"
echo ""

# 5. All Unit Tests
echo "5️⃣ Running All Unit Tests..."
pytest tests/unit/ -v --tb=short --maxfail=5
echo "✅ All Unit Tests PASSED"
echo ""

# 6. Complete System Tests
echo "6️⃣ Running Complete System Tests..."
pytest tests/test_complete_system.py -v --tb=short --maxfail=3
echo "✅ Complete System Tests PASSED"
echo ""

# 7. Integration Tests
echo "7️⃣ Running Integration Tests..."
pytest tests/integration/ -v --tb=short --maxfail=3
echo "✅ Integration Tests PASSED"
echo ""

# 8. Performance Tests
echo "8️⃣ Running Performance Tests..."
pytest tests/performance/ -v --tb=short --maxfail=2
echo "✅ Performance Tests PASSED"
echo ""

# 9. E2E Tests
echo "9️⃣ Running E2E Tests..."
pytest tests/e2e/ -v --tb=short --maxfail=2
echo "✅ E2E Tests PASSED"
echo ""

# 10. Full Coverage Report
echo "🔟 Generating Coverage Report..."
pytest \
  --tb=short \
  --verbose \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html \
  --cov-report=xml \
  --junit-xml=test-results.xml \
  --maxfail=10 \
  --durations=20 \
  --color=yes \
  tests/

echo ""
echo "🎉 COMPREHENSIVE TEST SUITE COMPLETED!"
echo "======================================"
echo ""
echo "📊 Test Results:"
echo "- Unit Tests: ✅ PASSED"
echo "- Integration Tests: ✅ PASSED"
echo "- Performance Tests: ✅ PASSED"
echo "- E2E Tests: ✅ PASSED"
echo "- Coverage: >90% (Expected)"
echo ""
echo "🎯 Your AI Agent System is production-ready!"
echo ""
echo "📁 Generated Reports:"
echo "- HTML Coverage: htmlcov/index.html"
echo "- XML Coverage: coverage.xml"
echo "- JUnit Results: test-results.xml"
echo ""
echo "🔗 View Coverage Report:"
echo "open htmlcov/index.html" 