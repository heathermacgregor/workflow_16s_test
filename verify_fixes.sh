#!/bin/bash
# Verification script for all applied fixes
# Run this to confirm all critical fixes are in place

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "=================================================="
echo "16S Enrichment Pipeline - Fix Verification"
echo "=================================================="
echo "Base directory: $SCRIPT_DIR"
echo ""

PASS=0
FAIL=0

# Test 1: O(n²) fix
echo -n "1. O(n²) ENA fix (enumerate instead of index)... "
if grep -q "for idx_pos, accession in enumerate(accessions)" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/api/sequence/ena/backfill.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 2: Cache logging removed
echo -n "2. Cache logging spam removed... "
if ! grep -q "Cache hit:" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/api/environmental_data/other/tools/cache.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 3: Weather API type check
echo -n "3. Weather API defensive type check... "
if grep -q "isinstance(key, str)" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/api/environmental_data/other/geo_enrichment.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 4: Weather API retry logic
echo -n "4. Weather API retry logic (exponential backoff)... "
if grep -q "retry_delay \* (2 \*\* attempt)" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/api/environmental_data/other/geo_enrichment.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 5: Weather API increased timeout
echo -n "5. Weather API timeout increased to 10s... "
if grep -q "timeout=10" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/api/environmental_data/other/geo_enrichment.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 6: Geochemical progress logging
echo -n "6. Geochemical progress logging... "
if grep -q "Geochemical progress" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/downstream/steps/backfill.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 7: Environmental enrichment type checking
echo -n "7. Environmental enrichment defensive type checking... "
if grep -q "flat_indices" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/api/environmental_data/other/main.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

# Test 8: Enhanced enrichment logging
echo -n "8. Enhanced enrichment logging (column composition)... "
if grep -q "JRC Water" "$SCRIPT_DIR/workflow_16s/src/workflow_16s/downstream/steps/backfill.py" 2>/dev/null; then
    echo "✅ PASS"
    ((PASS++))
else
    echo "❌ FAIL"
    ((FAIL++))
fi

echo ""
echo "=================================================="
echo "Results: $PASS passed, $FAIL failed"
echo "=================================================="

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "✅ ALL FIXES VERIFIED - Pipeline is ready for testing!"
    echo ""
    exit 0
else
    echo ""
    echo "❌ Some fixes are missing - review implementation"
    echo ""
    exit 1
fi
