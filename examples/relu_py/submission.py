#!POPCORN leaderboard relu_py

from task import input_t, output_t
from reference import ref_kernel  # Import for placeholder implementation

def custom_kernel(data: input_t) -> output_t:
    """
    Custom kernel implementation for the ReLU activation operation.

    Args:
        data: The input tensor.

    Returns:
        The result of the ReLU operation.
    """

    # TODO: Implement your optimized CUDA kernel here.
    # Placeholder implementation using the reference kernel:
    return ref_kernel(data)


# Example usage (for testing - remove or comment out for submission):
if __name__ == '__main__':
    import torch
    from reference import generate_input
    m, n, k = 3, 4, 5
    seed = 42
    input_data = generate_input(m, n, k, seed)
    output_data = custom_kernel(input_data)
    print(output_data) # Or any other testing/verification you want
