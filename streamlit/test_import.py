#!/usr/bin/env python3
"""Test script to debug import issues"""

import sys
from pathlib import Path

print("Current working directory:", Path.cwd())
print("Script location (__file__):", __file__)
print("Script path:", Path(__file__))
print("Parent:", Path(__file__).parent)
print("Parent.parent:", Path(__file__).parent.parent)
print("Parent.parent.resolve():", Path(__file__).parent.parent.resolve())

parent_dir = Path(__file__).parent.parent.resolve()
print("\nAdding to sys.path:", parent_dir)
sys.path.insert(0, str(parent_dir))

print("\nsys.path[0:3]:")
for i, p in enumerate(sys.path[:3]):
    print(f"  {i}: {p}")

print("\nLooking for feature_pipeline.py in parent_dir:")
feature_pipeline_path = parent_dir / "feature_pipeline.py"
print(f"  Path: {feature_pipeline_path}")
print(f"  Exists: {feature_pipeline_path.exists()}")

print("\nAttempting import...")
try:
    import feature_pipeline
    print("✓ SUCCESS: feature_pipeline imported")
    print(f"  Module location: {feature_pipeline.__file__}")
    from feature_pipeline import FeatureReproducer
    print("✓ SUCCESS: FeatureReproducer imported")
except ImportError as e:
    print(f"✗ FAILED: {e}")
    print("\nFiles in parent_dir:")
    for f in sorted(parent_dir.glob("*.py")):
        print(f"  - {f.name}")
