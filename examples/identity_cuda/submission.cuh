#include <array>
#include <vector>
#include "reference.cuh"

// checks that a CUDA API call returned successfully, otherwise prints an error message and exits.
static void cuda_check(cudaError_t status, const char* expr, const char* file, int line, const char* function);
#define cuda_check_(expr) cuda_check(expr, #expr, __FILE__, __LINE__, __FUNCTION__)

__global__ void copy_kernel(float *input, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        output[idx] = input[idx];
    }
}

output_t custom_kernel(input_t data)
{
    output_t result;

    for (int i = 0; i < N_SIZES; ++i)
    {
        int N = Ns[i];
        std::cout << "HANDLING SIZE "  <<  i << ": " << N << "\n";
        result[i].resize(N);

        // Allocate device memory
        float *d_input, *d_output;
        cuda_check_(cudaMalloc(&d_input, N * sizeof(float)));
        cuda_check_(cudaMalloc(&d_output, N * sizeof(float)));

        // Copy input to device
        cuda_check_(cudaMemcpy(d_input, data[i].data(), N * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel
        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        copy_kernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
        cuda_check_(cudaGetLastError());

        // Copy result back to host
        cuda_check_(cudaMemcpy(result[i].data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        cuda_check_(cudaFree(d_input));
        cuda_check_(cudaFree(d_output));
    }

    return result;
}
