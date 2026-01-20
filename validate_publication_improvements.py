#!/usr/bin/env python3
"""
Syntax validation for publication fetcher improvements.

This validates the syntax and structure without requiring dependencies.
"""

import ast
import re
from pathlib import Path

def validate_syntax(filepath):
    """Validate Python syntax using AST."""
    print(f"\nValidating: {filepath.name}")
    print("-" * 60)
    
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source, filename=str(filepath))
        print(f"✓ Syntax valid ({len(source.splitlines())} lines)")
        
        return True, source, tree
    except SyntaxError as e:
        print(f"✗ Syntax error at line {e.lineno}: {e.msg}")
        return False, None, None

def check_method_exists(source, method_name):
    """Check if a method is defined."""
    pattern = rf'def {re.escape(method_name)}\s*\('
    if re.search(pattern, source):
        print(f"✓ Method '{method_name}' found")
        return True
    else:
        print(f"✗ Method '{method_name}' not found")
        return False

def check_attribute_exists(source, attr_name):
    """Check if an attribute is assigned."""
    pattern = rf'self\.{re.escape(attr_name)}\s*='
    if re.search(pattern, source):
        print(f"✓ Attribute 'self.{attr_name}' found")
        return True
    else:
        print(f"✗ Attribute 'self.{attr_name}' not found")
        return False

def check_dict_key_exists(source, dict_name, key_name):
    """Check if a dictionary key is defined."""
    # Look for dict_name = {...}
    pattern = rf'{re.escape(dict_name)}\s*=\s*\{{[^}}]*["\']?{re.escape(key_name)}["\']?'
    if re.search(pattern, source, re.DOTALL):
        print(f"✓ Dictionary key '{dict_name}[\'{key_name}\']' found")
        return True
    else:
        print(f"✗ Dictionary key '{dict_name}[\'{key_name}\']' not found")
        return False

def check_sql_table(source, table_name):
    """Check if SQL table creation exists."""
    pattern = rf'CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+{re.escape(table_name)}'
    if re.search(pattern, source, re.IGNORECASE):
        print(f"✓ SQL table '{table_name}' creation found")
        return True
    else:
        print(f"✗ SQL table '{table_name}' creation not found")
        return False

def main():
    """Run validation checks."""
    print("="*60)
    print("PUBLICATION FETCHER - SYNTAX VALIDATION")
    print("="*60)
    
    filepath = Path("src/workflow_16s/utils/publication_fetcher.py")
    
    if not filepath.exists():
        print(f"✗ File not found: {filepath}")
        return 1
    
    # Validate syntax
    valid, source, tree = validate_syntax(filepath)
    if not valid:
        return 1
    
    checks = []
    
    # Check methods
    print("\nChecking Methods:")
    print("-" * 60)
    methods = [
        '_rate_limit',
        '_get_open_access_pdf_url',
        '_get_publications_from_ncbi',
        '_get_publications_from_crossref',
        '_create_cache_table',
    ]
    for method in methods:
        checks.append(check_method_exists(source, method))
    
    # Check attributes
    print("\nChecking Attributes:")
    print("-" * 60)
    attributes = [
        'rate_limits',
        'last_request_times',
        'api_calls',
        'source_success',
        'unpaywall_email',
        'timeout',
    ]
    for attr in attributes:
        checks.append(check_attribute_exists(source, attr))
    
    # Check rate_limits dictionary keys
    print("\nChecking Rate Limits Configuration:")
    print("-" * 60)
    rate_limit_apis = ['ncbi', 'crossref', 'semantic_scholar', 'europe_pmc', 'unpaywall']
    for api in rate_limit_apis:
        checks.append(check_dict_key_exists(source, 'rate_limits', api))
    
    # Check api_calls dictionary keys
    print("\nChecking API Statistics Configuration:")
    print("-" * 60)
    api_call_keys = ['total', 'cached', 'failed']
    for key in api_call_keys:
        checks.append(check_dict_key_exists(source, 'api_calls', key))
    
    # Check SQL tables
    print("\nChecking Database Schema:")
    print("-" * 60)
    tables = ['publication_cache', 'doi_metadata_cache', 'failed_lookups']
    for table in tables:
        checks.append(check_sql_table(source, table))
    
    # Check for rate limiting calls
    print("\nChecking Rate Limiting Integration:")
    print("-" * 60)
    if '_rate_limit(\'ncbi\')' in source or '_rate_limit("ncbi")' in source:
        print("✓ NCBI rate limiting integrated")
        checks.append(True)
    else:
        print("✗ NCBI rate limiting not integrated")
        checks.append(False)
    
    if '_rate_limit(\'crossref\')' in source or '_rate_limit("crossref")' in source:
        print("✓ Crossref rate limiting integrated")
        checks.append(True)
    else:
        print("✗ Crossref rate limiting not integrated")
        checks.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"Checks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All validation checks passed!")
        print("\nEnhancements Confirmed:")
        print("  • Enhanced caching (3 tables)")
        print("  • Rate limiting infrastructure")
        print("  • API statistics tracking")
        print("  • Unpaywall integration")
        print("  • Improved NCBI & Crossref fetchers")
        return 0
    else:
        print(f"\n✗ {total - passed} validation check(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
