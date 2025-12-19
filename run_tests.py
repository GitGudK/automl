#!/usr/bin/env python3
"""
Test runner for all AutoML tests

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py app          # Run app tests only
    python run_tests.py pipeline     # Run feature pipeline tests only
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests import test_app, test_feature_pipeline


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)

    # Run app tests
    print("\n" + "="*80)
    print("APP TESTS")
    print("="*80)
    os.system('python tests/test_app.py')

    # Run feature pipeline tests
    print("\n\n" + "="*80)
    print("FEATURE PIPELINE TESTS")
    print("="*80)
    os.system('python tests/test_feature_pipeline.py')


def run_app_tests():
    """Run app tests only"""
    os.system('python tests/test_app.py')


def run_pipeline_tests():
    """Run feature pipeline tests only"""
    os.system('python tests/test_feature_pipeline.py')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == 'app':
            run_app_tests()
        elif test_type == 'pipeline':
            run_pipeline_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("\nUsage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py app          # Run app tests only")
            print("  python run_tests.py pipeline     # Run feature pipeline tests only")
            sys.exit(1)
    else:
        run_all_tests()
