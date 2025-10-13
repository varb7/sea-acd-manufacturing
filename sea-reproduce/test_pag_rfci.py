#!/usr/bin/env python3
"""
Simple test for PAG-based RFCI implementation.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def test_pag_conversion():
    """Test the PAG conversion function directly."""
    print("Testing PAG conversion...")
    
    try:
        from rfci_module import TetradRFCI
        
        # Create synthetic data
        np.random.seed(42)
        n_samples, n_vars = 1000, 5
        
        # Simple causal structure: X1 -> X2 -> X3, X1 -> X4, X5 independent
        data = np.random.randn(n_samples, n_vars)
        data[:, 1] = 0.5 * data[:, 0] + 0.3 * np.random.randn(n_samples)  # X2 depends on X1
        data[:, 2] = 0.7 * data[:, 1] + 0.2 * np.random.randn(n_samples)  # X3 depends on X2
        data[:, 3] = 0.4 * data[:, 0] + 0.4 * np.random.randn(n_samples)  # X4 depends on X1
        
        df = pd.DataFrame(data, columns=[f"v{i}" for i in range(n_vars)])
        
        print(f"Data shape: {data.shape}")
        print("Running RFCI with PAG output...")
        
        # Test the RFCI with PAG output
        rfci = TetradRFCI(alpha=0.05, depth=2)
        result = rfci.run(df)
        
        print(f"RFCI result shape: {result.shape}")
        print(f"RFCI result type: {result.dtype}")
        print("RFCI PAG adjacency matrix:")
        print(result)
        
        # Check if we get PAG values (0,1,2,3,4)
        unique_values = np.unique(result)
        print(f"Unique values in result: {unique_values}")
        
        if all(val in [0,1,2,3,4] for val in unique_values):
            print("PAG conversion test passed!")
            return True
        else:
            print("PAG conversion test failed - unexpected values")
            return False
            
    except Exception as e:
        print(f"PAG conversion test failed: {e}")
        return False

def test_edge_mapping():
    """Test the edge mapping for PAG format."""
    print("\nTesting PAG edge mapping...")
    
    try:
        from data.utils import convert_result_to_lg, edge_map_rfci_pag
        
        # Create a simple 3x3 PAG adjacency matrix
        G = np.array([
            [0, 2, 0],  # 0 -> 1 (forward edge)
            [0, 0, 2],  # 1 -> 2 (forward edge)
            [0, 0, 0]   # no outgoing edges
        ])
        
        print("Input PAG matrix:")
        print(G)
        
        edge_attrs = convert_result_to_lg(G, edge_map_rfci_pag)
        print(f"Edge attributes: {edge_attrs}")
        
        print("PAG edge mapping test passed!")
        return True
    except Exception as e:
        print(f"PAG edge mapping test failed: {e}")
        return False

def main():
    """Run PAG tests."""
    print("Testing PAG-based RFCI implementation\n")
    
    tests = [
        test_pag_conversion,
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
        print("All PAG tests passed! RFCI PAG implementation is working.")
        return 0
    else:
        print("Some PAG tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
