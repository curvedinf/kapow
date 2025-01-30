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

def test_approximate_sqrt():
    # Loop through 1000 iterations with random integers
    for _ in range(1000):
        random_int = random.randint(0, 100000)
        approx_sqrt = approximate_sqrt(random_int)
        actual_sqrt = math.sqrt(random_int)
        diff = abs(approx_sqrt - actual_sqrt)
        
        print(f"Square root of {random_int}:")
        print(f"Approx: {approx_sqrt}")
        print(f"Actual: {actual_sqrt}")
        print(f"Diff: {diff}")
    
    # Run a final iteration with a new random integer to assert tolerance
    test_case = random.randint(0, 100000)
    approx_sqrt = approximate_sqrt(test_case)
    actual_sqrt = math.sqrt(test_case)
    diff = abs(approx_sqrt - actual_sqrt)

    tolerance = 1e-6  # Define a small tolerance for floating point comparisons
    assert diff < tolerance, f"Difference {diff} exceeds tolerance {tolerance}"