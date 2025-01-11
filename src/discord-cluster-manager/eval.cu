#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>
#include <memory>

#include "reference.cuh"
#include "train.cuh"

#define WARMUP_RUNS 10
#define TIMED_RUNS 100

struct Closer {
    void operator()(std::FILE* file) {
        std::fclose(file);
    }
};

struct PopcornOutput {
    template<class... Args>
    void printf(Args&&... args) {
        ::fprintf(File.get(), std::forward<Args>(args)...);
    }

    void log(const char* key, const char* value) {
        printf("%s: %s\n", key, value);
    }

    template<class T>
    void log(const char* key, T&& value) {
        log(key, std::to_string(value).c_str());
    }

    std::unique_ptr<std::FILE, Closer> File;
};

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

double measure_runtime() {
    std::cout << "warming up..." << std::endl;

    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto data = generate_input();
        // discard result; this is just warmup, we don't care what it returns
        (void)custom_kernel(data);
    }
    cuda_check(cudaDeviceSynchronize());

    std::vector<std::int64_t> durations;
    durations.reserve(TIMED_RUNS);

    for (int i = 0; i < TIMED_RUNS; i++) {
        auto data = generate_input();
        // make a copy of the input data to be used by the reference implementation
        auto copy = data;

        auto start = std::chrono::high_resolution_clock::now();
        // move data into custom_kernel, so that if custom_kernel takes large std::vectors or similar by value,
        // we're not measuring the copy overhead.
        auto submission_output = custom_kernel(std::move(data));
        cuda_check(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        auto reference_output = ref_kernel(copy);
        if (!check_implementation(submission_output, reference_output)) {
            return -1.0;    // negative result indicates fail
        }

    }

    // calculate duration statistics
    std::int64_t total_duration = std::accumulate(durations.begin(), durations.end(), (std::int64_t)0);

    double average_duration = (double)total_duration / 1e9 / TIMED_RUNS;
    std::cout << "submitted kernel runtime: " << average_duration << " seconds" << std::endl;
    return average_duration;
}

int main() {
    const char *output_fd = std::getenv("POPCORN_FD");
    PopcornOutput logger;
    if (output_fd) {
        int fd = std::stoi(output_fd);
        logger.File.reset(::fdopen(fd, "w"));
    } else {
        return 4;       // pytest: usage error
    }

    auto data = generate_input();
    auto reference_output = ref_kernel(data);
    auto submission_output = custom_kernel(data);

    if (!check_implementation(submission_output, reference_output)) {
        logger.log("check", "fail");
        return 1;
    }

    double s = measure_runtime();
    if (s < 0) {
        logger.log("check", "fail");
        return 1;
    }

    logger.log("check", "pass");
    logger.log("score", s);

    return 0;
}
