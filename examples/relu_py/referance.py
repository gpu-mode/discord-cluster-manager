import torch
from task import input_t, output_t
from utils import verbose_allclose

def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    """Generates input data for the ReLU activation kernel.

    Args:
        m: The first dimension of the tensor.
        n: The second dimension of the tensor.
        k: The third dimension of the tensor.
        seed: The seed for random number generation.

    Returns:
        The input tensor as an input_t.
    """
    torch.manual_seed(seed)
    # Generate on CUDA if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand(m, n, k, device=device) * 2 - 1  # Values between -1 and 1
    return input_tensor.half() # float16


def ref_kernel(data: input_t) -> output_t:
    """Provides a reference implementation of the ReLU activation kernel.

    Args:
        data: The input tensor.

    Returns:
        The result of the ReLU operation as an output_t.
    """
    return torch.relu(data)


def check_implementation(data: input_t, output: output_t) -> str:
    """Checks if a custom kernel implementation is correct.

    Args:
        data: The input tensor.
        output: The output from a custom kernel.

    Returns:
        An error string describing the mismatch, or an empty string if correct.
    """
    expected = ref_kernel(data)
    match, diff_indices = verbose_allclose(output, expected)

    if not match:
        error_string = "Mismatch found at indices: " + str(diff_indices)
        return error_string
    else:
        return ""

if __name__ == '__main__':
    # Example usage (for testing)
    m, n, k = 3, 4, 5
    seed = 42
    input_data = generate_input(m, n, k, seed)
    ref_output = ref_kernel(input_data)

    # Example Check
    # (Assuming you have a dummy custom output for testing check_implementation)
    dummy_output = ref_output.clone() # Create a copy for testing
    dummy_output[0,0,0] = 100 # Introduce a difference for testing mismatch
    error_message = check_implementation(input_data, dummy_output)
    if error_message:
        print(f"Error: {error_message}")
    else:
        print("Implementation is correct.")

    correct_check = check_implementation(input_data, ref_output)
    if correct_check:
        print(f"Error: {correct_check}")
    else:
        print("Correct Implementation check passed")
