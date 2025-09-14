#!/usr/bin/env python3
"""Simple test runner for smoke tests without pytest dependency."""

import sys
import subprocess
from pathlib import Path

def run_smoke_tests():
    """Run all smoke tests and report results."""
    print("🧪 Running PyContextify Smoke Tests")
    print("=" * 50)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    tests = [
        script_dir / "test_mcp_server.py",
        script_dir / "test_mcp_functionality.py"
    ]
    
    results = {}
    
    for test in tests:
        test_path = Path(test)
        if not test_path.exists():
            print(f"❌ Test file not found: {test}")
            results[test] = False
            continue
        
        print(f"\n🏃 Running {test_path.name}...")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=False,  # Show output directly
                check=True
            )
            results[test] = True
            print(f"✅ {test_path.name} passed")
            
        except subprocess.CalledProcessError as e:
            results[test] = False
            print(f"❌ {test_path.name} failed with exit code {e.returncode}")
        except Exception as e:
            results[test] = False
            print(f"❌ {test_path.name} failed with error: {e}")
    
    # Summary
    print("\n📊 Smoke Test Results")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {Path(test).name}")
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All smoke tests passed!")
        return True
    else:
        print("⚠️  Some smoke tests failed")
        return False

def main():
    """Main entry point."""
    success = run_smoke_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()