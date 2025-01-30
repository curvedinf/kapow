import random
import math
import pytest
from kapow import mf

def optimal_sqrt(input=float):
    return math.sqrt(input)

@mf(optimal_sqrt)
def approximate_sqrt(input=float):
    # This function is a stub for metafunction optimization, its logic is irrelevant here
    return float(input)**0.5

@pytest.mark.parametrize("test_case", [random.randint(0, 100000) for _ in range(10)])
def test_approximate_sqrt(test_case):
    approx_sqrt = approximate_sqrt(test_case)
    actual_sqrt = math.sqrt(test_case)
    diff = abs(approx_sqrt - actual_sqrt)
    
    print(f"Square root of {test_case}:")
    print(f"Approx: {approx_sqrt}")
    print(f"Actual: {actual_sqrt}")
    print(f"Diff: {diff}")
    
    tolerance = 1e-6  # Define a small tolerance for floating point comparisons
    assert diff < tolerance, f"Difference {diff} exceeds tolerance {tolerance}"
