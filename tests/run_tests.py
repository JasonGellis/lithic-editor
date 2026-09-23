#!/usr/bin/env python
"""
Convenience script to run tests with coverage.
"""

import sys
import subprocess
from importlib.util import find_spec
from pathlib import Path

def run_tests():
    """Run pytest with coverage reporting."""
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Basic test command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Shorter traceback format
    ]
    
    # Add coverage if pytest-cov is installed
    has_cov = find_spec("pytest_cov") is not None
    if has_cov:
        cmd.extend([
            "--cov=lithic_editor",
            "--cov-report=term-missing",
            "--cov-report=html",
        ])
        print("Running tests with coverage...")
    else:
        print("Running tests without coverage (install pytest-cov for coverage reports)")
    
    # Add color if supported
    cmd.append("--color=yes")
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        if has_cov:
            print("📊 Coverage report saved to htmlcov/index.html")
    else:
        print("\n❌ Some tests failed")
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())