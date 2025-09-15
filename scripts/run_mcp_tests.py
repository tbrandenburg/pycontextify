#!/usr/bin/env python3
"""
Simple script to run PyContextify MCP server tests.

This script provides a clean way to run the MCP end-to-end tests
with proper reporting and coverage information.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_mcp_tests():
    """Run the comprehensive MCP test suite."""
    print("🚀 PyContextify MCP Server Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Check if test file exists
    test_file = Path("tests/test_mcp_simple.py")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    print(f"📁 Running tests from: {test_file}")
    print(f"📊 Testing all 6 MCP functions with multiple document types\n")
    
    # Run pytest with comprehensive options
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "-v",                    # Verbose output
        "--tb=short",           # Shorter traceback format
        "--cov=pycontextify",   # Coverage reporting
        "--cov-report=term-missing",  # Show missing lines
        "--durations=5",        # Show 5 slowest tests
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, text=True, capture_output=False)
        
        elapsed = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"⏱️  Total runtime: {elapsed:.1f} seconds")
        
        if result.returncode == 0:
            print("🎉 All MCP tests passed successfully!")
            
            # Summary of what was tested
            print(f"\n📋 Test Summary:")
            print(f"   ✅ status() - System status reporting")
            print(f"   ✅ index_document() - Single file indexing") 
            print(f"   ✅ index_code() - Codebase directory indexing")
            print(f"   ✅ search() - Basic semantic search")
            print(f"   ✅ search_with_context() - Enhanced search")
            print(f"   ✅ Error handling - Invalid input testing")
            print(f"   ✅ Full workflow - End-to-end pipeline")
            print(f"   ✅ Function availability - Direct access verification")
            
            print(f"\n📄 Document types tested:")
            print(f"   • Markdown (.md) - Documentation, guides")
            print(f"   • Text (.txt) - Code files, configs, general content")
            print(f"   • Codebase indexing - Multi-file directory processing")
            
            return True
        else:
            print(f"❌ Tests failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_quick_smoke_test():
    """Run just a quick smoke test for faster feedback."""
    print("💨 Quick Smoke Test")
    print("-" * 30)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_mcp_simple.py::TestMCPFunctions::test_status_function",
        "tests/test_mcp_simple.py::TestMCPFunctions::test_index_document_function", 
        "-v", "--tb=short"
    ]
    
    result = subprocess.run(cmd, text=True, capture_output=False)
    return result.returncode == 0


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        # Quick smoke test
        success = run_quick_smoke_test()
    else:
        # Full test suite
        success = run_mcp_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()