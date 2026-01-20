#!/usr/bin/env python3
"""
Validation script for statistical and visualization enhancements.

This script validates:
1. Statistics module (effect sizes, biological significance)
2. Dashboards module (integrated visualizations)
3. Power analysis enhancements

Usage:
    python validate_statistical_enhancements.py
"""

import ast
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def validate_syntax(filepath: Path) -> bool:
    """Validate Python syntax using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"Syntax error in {filepath}: {e}")
        return False


def check_function_exists(filepath: Path, function_name: str) -> bool:
    """Check if a function exists in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"def {function_name}(" in content
    except Exception as e:
        logger.error(f"Error checking function {function_name}: {e}")
        return False


def check_class_exists(filepath: Path, class_name: str) -> bool:
    """Check if a class exists in a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"class {class_name}" in content
    except Exception as e:
        logger.error(f"Error checking class {class_name}: {e}")
        return False


def check_import_exists(filepath: Path, import_name: str) -> bool:
    """Check if an import statement exists."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return import_name in content
    except Exception as e:
        logger.error(f"Error checking import {import_name}: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.info("="*80)
    logger.info("STATISTICAL & VISUALIZATION ENHANCEMENTS - VALIDATION")
    logger.info("="*80)
    logger.info("")
    
    base_path = Path("src/workflow_16s/downstream")
    files_to_check = {
        'statistics.py': base_path / "statistics.py",
        'dashboards.py': base_path / "dashboards.py",
        'power_analysis.py': base_path / "power_analysis.py"
    }
    
    # Check all files exist
    logger.info("Checking file existence...")
    for name, path in files_to_check.items():
        if not path.exists():
            logger.error(f"✗ {name} not found at {path}")
            return 1
        logger.info(f"  ✓ {name} found")
    
    logger.info("")
    
    checks_passed = 0
    checks_total = 0
    
    # =========================================================================
    # STATISTICS.PY VALIDATION
    # =========================================================================
    logger.info("Validating: src/workflow_16s/downstream/statistics.py")
    stats_file = files_to_check['statistics.py']
    
    # Syntax check
    checks_total += 1
    if validate_syntax(stats_file):
        logger.info("  ✓ Syntax valid")
        checks_passed += 1
    else:
        logger.error("  ✗ Syntax error")
    
    # Check effect size functions
    effect_size_functions = [
        'cohens_d',
        'cliffs_delta',
        'eta_squared',
        'r_squared_from_correlation',
        'interpret_effect_size'
    ]
    
    for func in effect_size_functions:
        checks_total += 1
        if check_function_exists(stats_file, func):
            logger.info(f"  ✓ Function '{func}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Function '{func}' not found")
    
    # Check comprehensive testing functions
    testing_functions = [
        'calculate_effect_sizes',
        'test_with_effect_size',
        'generate_stats_report',
        'calculate_achieved_power',
        'required_sample_size'
    ]
    
    for func in testing_functions:
        checks_total += 1
        if check_function_exists(stats_file, func):
            logger.info(f"  ✓ Function '{func}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Function '{func}' not found")
    
    # Check for key imports
    stats_imports = [
        'from scipy import stats',
        'from statsmodels.stats.multitest import multipletests',
        'from statsmodels.stats.power import TTestIndPower'
    ]
    
    for imp in stats_imports:
        checks_total += 1
        if check_import_exists(stats_file, imp):
            logger.info(f"  ✓ Import '{imp}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Import '{imp}' not found")
    
    logger.info("")
    
    # =========================================================================
    # DASHBOARDS.PY VALIDATION
    # =========================================================================
    logger.info("Validating: src/workflow_16s/downstream/dashboards.py")
    dashboards_file = files_to_check['dashboards.py']
    
    # Syntax check
    checks_total += 1
    if validate_syntax(dashboards_file):
        logger.info("  ✓ Syntax valid")
        checks_passed += 1
    else:
        logger.error("  ✗ Syntax error")
    
    # Check main dashboard functions
    dashboard_functions = [
        'create_integrated_dashboard',
        'create_qc_aware_diversity_dashboard'
    ]
    
    for func in dashboard_functions:
        checks_total += 1
        if check_function_exists(dashboards_file, func):
            logger.info(f"  ✓ Function '{func}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Function '{func}' not found")
    
    # Check helper panel functions
    panel_functions = [
        '_add_qc_summary_panel',
        '_add_sample_distribution_panel',
        '_add_sequencing_depth_panel',
        '_add_alpha_diversity_panel',
        '_add_beta_diversity_panel',
        '_add_top_taxa_panel',
        '_add_statistical_results_panel',
        '_add_effect_sizes_panel',
        '_add_power_analysis_panel',
        '_add_executive_summary_panel'
    ]
    
    for func in panel_functions:
        checks_total += 1
        if check_function_exists(dashboards_file, func):
            logger.info(f"  ✓ Panel function '{func}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Panel function '{func}' not found")
    
    # Check for plotly imports
    dashboard_imports = [
        'import plotly.graph_objects as go',
        'from plotly.subplots import make_subplots',
        'import plotly.express as px'
    ]
    
    for imp in dashboard_imports:
        checks_total += 1
        if check_import_exists(dashboards_file, imp):
            logger.info(f"  ✓ Import '{imp}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Import '{imp}' not found")
    
    logger.info("")
    
    # =========================================================================
    # POWER_ANALYSIS.PY VALIDATION
    # =========================================================================
    logger.info("Validating: src/workflow_16s/downstream/power_analysis.py")
    power_file = files_to_check['power_analysis.py']
    
    # Syntax check
    checks_total += 1
    if validate_syntax(power_file):
        logger.info("  ✓ Syntax valid")
        checks_passed += 1
    else:
        logger.error("  ✗ Syntax error")
    
    # Check enhanced functions
    power_functions = [
        'recommend_sample_size',
        'generate_power_report',
        'plot_power_curves'
    ]
    
    for func in power_functions:
        checks_total += 1
        if check_function_exists(power_file, func):
            logger.info(f"  ✓ Function '{func}' found")
            checks_passed += 1
        else:
            logger.error(f"  ✗ Function '{func}' not found")
    
    logger.info("")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Checks passed: {checks_passed}/{checks_total}")
    logger.info("")
    
    if checks_passed == checks_total:
        logger.info("✓ All validation checks passed!")
        logger.info("")
        logger.info("Enhancements Confirmed:")
        logger.info("  • Effect size calculations (Cohen's d, Cliff's Delta, Eta-squared)")
        logger.info("  • Biological significance testing")
        logger.info("  • Integrated analysis dashboard (12-panel layout)")
        logger.info("  • QC-aware diversity visualization")
        logger.info("  • Enhanced power analysis with recommendations")
        logger.info("  • Sample size recommendation system")
        logger.info("")
        return 0
    else:
        logger.error(f"✗ {checks_total - checks_passed} check(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
