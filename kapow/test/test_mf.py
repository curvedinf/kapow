import random
import math
from kapow import mf

def optimal_sqrt(mf_output, args, kwargs):
    return math.sqrt(kwargs["input_number"])

@mf(optimal_sqrt)
def approximate_sqrt(input_number=1.0):
    return 1.0

def test_approximate_sqrt():
    # Loop through 1000 iterations with random integers
    for _ in range(1000):
        random_int = random.randint(0, 100000)
        approx_sqrt = approximate_sqrt(input_number=random_int)
        actual_sqrt = math.sqrt(random_int)
        diff = abs(approx_sqrt - actual_sqrt)
        
        print(f"Square root of {random_int}:")
        print(f"Approx: {approx_sqrt}")
        print(f"Actual: {actual_sqrt}")
        print(f"Diff: {diff}")
    
    # Run a final iteration with a new random integer to assert tolerance
    test_case = random.randint(0, 100000)
    approx_sqrt = approximate_sqrt(input_number=test_case)
    actual_sqrt = math.sqrt(test_case)
    diff = abs(approx_sqrt - actual_sqrt)

    tolerance = 1e-6  # Define a small tolerance for floating point comparisons
    assert diff < tolerance, f"Difference {diff} exceeds tolerance {tolerance}"