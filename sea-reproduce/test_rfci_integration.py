#!/usr/bin/env python3
"""
Test script to verify RFCI integration with SEA pipeline.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def test_rfci_wrapper():
    """Test the RFCI wrapper function."""
    from data.utils import run_rfci
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_vars = 1000, 5
    
    # Simple causal structure: X1 -> X2 -> X3, X1 -> X4, X5 independent
    data = np.random.randn(n_samples, n_vars)
    data[:, 1] = 0.5 * data[:, 0] + 0.3 * np.random.randn(n_samples)  # X2 depends on X1
    data[:, 2] = 0.7 * data[:, 1] + 0.2 * np.random.randn(n_samples)  # X3 depends on X2
    data[:, 3] = 0.4 * data[:, 0] + 0.4 * np.random.randn(n_samples)  # X4 depends on X1
    
    print("Testing RFCI wrapper...")
    print(f"Data shape: {data.shape}")
    
    try:
        # Test the wrapper
        result = run_rfci(data, alpha=0.05, depth=2)
        
        if result is not None:
            print(f"RFCI result shape: {result.shape}")
            print(f"RFCI result type: {result.dtype}")
            print("RFCI adjacency matrix:")
            print(result)
            print("RFCI wrapper test passed!")
            return True
        else:
            print("RFCI returned None")
            return False
            
    except Exception as e:
        print(f"RFCI wrapper test failed: {e}")
        return False

def test_algorithm_selector():
    """Test that RFCI is properly registered in the algorithm selector."""
    from data.dataset import get_run_alg
    
    print("\nTesting algorithm selector...")
    
    try:
        rfci_func = get_run_alg("rfci")
        print(f"RFCI function: {rfci_func}")
        print("Algorithm selector test passed!")
        return True
    except Exception as e:
        print(f"Algorithm selector test failed: {e}")
        return False

def test_edge_mapping():
    """Test the edge mapping for RFCI results."""
    from data.utils import convert_result_to_lg, edge_map_rfci_bin
    
    print("\nTesting edge mapping...")
    
    try:
        # Create a simple 3x3 adjacency matrix
        G = np.array([
            [0, 1, 0],  # 0 -> 1
            [0, 0, 1],  # 1 -> 2  
            [0, 0, 0]   # no outgoing edges
        ])
        
        edge_attrs = convert_result_to_lg(G, edge_map_rfci_bin)
        print(f"Edge attributes: {edge_attrs}")
        print("Edge mapping test passed!")
        return True
    except Exception as e:
        print(f"Edge mapping test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing RFCI integration with SEA pipeline\n")
    
    tests = [
        test_rfci_wrapper,
        test_algorithm_selector, 
        test_edge_mapping
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! RFCI integration is ready.")
        return 0
    else:
        print("Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
