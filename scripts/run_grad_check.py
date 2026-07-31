"""
Reproduction script for Part 1 gradient-check tests.
Run via: python scripts/run_grad_check.py
"""
import sys
import pytest

if __name__ == "__main__":
    print("=" * 60)
    print(" Running AQUA Autodiff Gradient Check Test Suite...")
    print("=" * 60)
    
    # Run pytest on tests/ directory
    exit_code = pytest.main(["-v", "tests/test_autodiff.py", "tests/test_masked.py"])
    
    if exit_code == 0:
        print("\n All gradient checks and masked weight tests PASSED successfully!")
    else:
        print("\n Gradient check test suite FAILED!")
        
    sys.exit(exit_code)
