import torch
import numpy as np
import sys
import os

# Assuming your functions are in a module called 'network'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import trimmed_mean_screen, median_screen

def test_trimmed_mean_screen():
    """
    Test the trimmed_mean_screen function with various scenarios
    """
    print("=== Testing trimmed_mean_screen ===")
    
    # Create a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test 1: Basic case with 1D tensors
    print("\nTest 1: Basic case with 1D tensors")
    params_list_1d = [
        [torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, device=device)],
        [torch.tensor([2, 3, 4, 5, 6], dtype=torch.float, device=device)],
        [torch.tensor([3, 4, 5, 6, 7], dtype=torch.float, device=device)],
        [torch.tensor([4, 5, 6, 7, 8], dtype=torch.float, device=device)],
        [torch.tensor([5, 6, 7, 8, 9], dtype=torch.float, device=device)]
    ]
    trim_param = 1
    result_1d = trimmed_mean_screen(params_list_1d, trim_param)
    
    print(f"Original parameters:")
    for i, params in enumerate(params_list_1d):
        print(f"  Node {i}: {params[0]}")
    
    print(f"Trimmed mean result (trim={trim_param}):")
    print(f"  {result_1d[0]}")
    
    # Test 2: Basic case with 2D tensors
    print("\nTest 2: Basic case with 2D tensors")
    params_list_2d = [
        [torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float, device=device)],
        [torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float, device=device)],
        [torch.tensor([[3, 4, 5], [6, 7, 8]], dtype=torch.float, device=device)],
        [torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.float, device=device)],
        [torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float, device=device)]
    ]
    trim_param = 1
    result_2d = trimmed_mean_screen(params_list_2d, trim_param)
    
    print(f"Original parameters:")
    for i, params in enumerate(params_list_2d):
        print(f"  Node {i}: {params[0]}")
    
    print(f"Trimmed mean result (trim={trim_param}):")
    print(f"  {result_2d[0]}")
    
    # Test 3: Edge case - not enough nodes for trimming
    print("\nTest 3: Edge case - not enough nodes for trimming")
    params_list_small = [
        [torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, device=device)],
        [torch.tensor([10, 20, 30, 40, 50], dtype=torch.float, device=device)]
    ]
    trim_param = 1
    print(f"Original parameters:")
    for i, params in enumerate(params_list_small):
        print(f"  Node {i}: {params[0]}")
    
    try:
        result_small = trimmed_mean_screen(params_list_small, trim_param)
        print(f"Result when not enough nodes (should fall back to median):")
        print(f"  {result_small[0]}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Test 4: Outlier handling
    print("\nTest 4: Outlier handling")
    params_list_outliers = [
        [torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, device=device)],
        [torch.tensor([2, 3, 4, 5, 6], dtype=torch.float, device=device)],
        [torch.tensor([3, 4, 5, 6, 7], dtype=torch.float, device=device)],
        [torch.tensor([4, 5, 6, 7, 8], dtype=torch.float, device=device)],
        [torch.tensor([100, 200, 300, 400, 500], dtype=torch.float, device=device)]  # Outlier
    ]
    trim_param = 1
    result_outliers = trimmed_mean_screen(params_list_outliers, trim_param)
    
    print(f"Original parameters:")
    for i, params in enumerate(params_list_outliers):
        print(f"  Node {i}: {params[0]}")
    
    print(f"Trimmed mean result (trim={trim_param}):")
    print(f"  {result_outliers[0]}")
    print(f"Regular mean (for comparison):")
    regular_mean = torch.stack([p[0] for p in params_list_outliers], dim=0).mean(dim=0)
    print(f"  {regular_mean}")
    
    # Test 5: Multiple parameter layers
    print("\nTest 5: Multiple parameter layers")
    params_list_multi = [
        [
            torch.tensor([1, 2, 3], dtype=torch.float, device=device),
            torch.tensor([[1, 2], [3, 4]], dtype=torch.float, device=device)
        ],
        [
            torch.tensor([2, 3, 4], dtype=torch.float, device=device),
            torch.tensor([[2, 3], [4, 5]], dtype=torch.float, device=device)
        ],
        [
            torch.tensor([3, 4, 5], dtype=torch.float, device=device),
            torch.tensor([[3, 4], [5, 6]], dtype=torch.float, device=device)
        ],
        [
            torch.tensor([4, 5, 6], dtype=torch.float, device=device),
            torch.tensor([[4, 5], [6, 7]], dtype=torch.float, device=device)
        ],
        [
            torch.tensor([5, 6, 7], dtype=torch.float, device=device),
            torch.tensor([[5, 6], [7, 8]], dtype=torch.float, device=device)
        ]
    ]
    trim_param = 1
    result_multi = trimmed_mean_screen(params_list_multi, trim_param)
    
    print(f"Original parameters (showing first layer only):")
    for i, params in enumerate(params_list_multi):
        print(f"  Node {i}: {params[0]}")
    
    print(f"Trimmed mean result (trim={trim_param}):")
    print(f"  Layer 1: {result_multi[0]}")
    print(f"  Layer 2: {result_multi[1]}")

if __name__ == "__main__":
    test_trimmed_mean_screen()