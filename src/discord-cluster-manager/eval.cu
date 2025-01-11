#include <chrono>
#include <iostream>

#include "reference.cuh"
#include "train.cuh"

#define WARMUP_RUNS 10
#define TIMED_RUNS 100

// checks that a CUDA API call returned successfully, otherwise prints an error message and exits.
static void cuda_check(cudaError_t status, const char* expr, const char* file, int line, const char* function)
{
    if(status != cudaSuccess) {
        std::cerr << "CUDA error (" << (int)status << ") while evaluating expression "
                  << expr << " at "
                  << file << '('
                  << line << ") in `"
                  << function << "`: "
                  << cudaGetErrorString(status) << std::endl;
        // following pytest convention, exit code 3 means internal error
        std::exit(3);
    }
}

#define cuda_check(expr) cuda_check(expr, #expr, __FILE__, __LINE__, __FUNCTION__)

float measure_runtime() {
    std::cout << "warming up..." << std::endl;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto data = generate_input();
        custom_kernel(data);
    }
    cuda_check(cudaDeviceSynchronize());

    using double_duration = std::chrono::duration<double>;
    double total_duration = 0.0;

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input();

        auto start = std::chrono::high_resolution_clock::now();
        auto submission_output = custom_kernel(data);
        cuda_check(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        total_duration += std::chrono::duration_cast<double_duration>(end - start).count();

        auto reference_output = ref_kernel(data);
        if (!check_implementation(submission_output, reference_output)) {
            std::cout << "check_implementation failed" << std::endl;
            return 1;
        }

    }


    double average_duration = total_duration / TIMED_RUNS;
    std::cout << "submitted kernel runtime: " << average_duration << " seconds" << std::endl;
    return average_duration;
}

int main() {
    auto data = generate_input();
    auto reference_output = ref_kernel(data);
    auto submission_output = custom_kernel(data);

    if (!check_implementation(submission_output, reference_output)) {
        std::cout << "check_implementation failed" << std::endl;
        return 1;
    }

    float s = measure_runtime();
    if (s < 0) {
        return 1;
    }

    std::cout << "score: " << s << std::endl;

    return 0;
}
