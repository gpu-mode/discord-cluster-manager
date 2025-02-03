#include <chrono>
#include <iostream>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <regex>
#include <cassert>
#include <tuple>
#include "utils.h"
#include "reference.cuh"

// forward declaration for user submission
output_t custom_kernel(input_t data);

#define WARMUP_RUNS 10
#define TIMED_RUNS 100

namespace
{
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

    void log(const std::string& key, const char* value) {
        printf("%s: %s\n", key.c_str(), value);
    }

    template<class T>
    void log(const std::string& key, T&& value) {
        log(key, std::to_string(value).c_str());
    }

    std::unique_ptr<std::FILE, Closer> File;
};

template<class F>
struct extract_signature_helper;

template<class R, class... Args>
struct extract_signature_helper<R(*)(Args...)> {
    using tuple_t = std::tuple<std::remove_const_t<std::remove_reference_t<Args>>...>;
};

struct TestCase {
    using Parameter = typename extract_signature_helper<decltype(&generate_input)>::tuple_t;
    static_assert(std::tuple_size<Parameter>() == ArgumentNames.size(), "Mismatch in argument name count");
    std::string spec;
    Parameter params;
};

template<class T>
void assign_value(T& target, const std::string& raw) {
    if constexpr (std::is_same_v<T, std::string>) {
        target = raw;
    } else {
        static_assert(std::is_same_v<T, int>, "Test arguments must be integers or strings");
        target = std::stoi(raw);
    }
}

template<std::size_t Index = ArgumentNames.size()>
bool set_param_value(TestCase::Parameter& param, const std::string& key, const std::string& raw, std::integral_constant<std::size_t, Index> = {}) {
    if(key == ArgumentNames[Index-1]) {
        assign_value(std::get<Index-1>(param), raw);
        return true;
    }
    if constexpr (Index != 1) {
        return set_param_value(param, key, raw, std::integral_constant<std::size_t, Index-1>{});
    }
    return false;
}

// reads a line from a std::FILE
std::string read_line(std::FILE* file) {
    std::string buf;
    while(true) {
        int next = std::fgetc(file);
        if (next == '\n' || next == EOF) {
            return buf;
        }
        buf.push_back((char)next);
    }

}

TestCase parse_test_case(const std::string& line) {
    // match a key-value pair of integer or string value
    static std::regex match_entry(R"(\s*([a-zA-Z]+):\s*([a-zA-Z]+|[+-]?[0-9]+)\s*)");

    TestCase tc;
    tc.spec = line;

    // split line into individual arguments
    std::vector<std::string> parts;
    std::size_t pos = 0;
    while(pos != std::string::npos) {
        std::size_t next = line.find(';', pos);
        parts.push_back(line.substr(pos, next));
        pos = next == std::string::npos ? next : next + 1;
    }

    // split arguments into kv pairs
    for(const std::string& arg : parts) {
        std::smatch m;
        if(!std::regex_match(arg, m, match_entry)) {
            std::cerr << "invalid test case: ''" << line << "'': '" << arg << "'" << std::endl;
            std::exit(ExitCodes::EXIT_TEST_SPEC);
        }

        // TODO check that we get all values
        // TODO check that no value is duplicate

        std::string key = std::string(m[1].first, m[1].second);
        std::string value = std::string(m[2].first, m[2].second);
        if(!set_param_value(tc.params, key, value)) {
            std::cerr << "invalid test case: ";
            std::cerr << "argument name '" << key << "' is invalid" << std::endl;
            std::exit(ExitCodes::EXIT_TEST_SPEC);
        }
    }
    return tc;
}

PopcornOutput open_logger() {
    PopcornOutput logger;
    const char *output_fd = std::getenv("POPCORN_FD");
    if (output_fd) {
        int fd = std::stoi(output_fd);
        logger.File.reset(::fdopen(fd, "w"));
    } else {
        std::cerr << "Missing POPCORN_FD file descriptor." << std::endl;
        std::exit(ExitCodes::EXIT_PIPE_FAIL);
    }
    unsetenv("POPCORN_FD");
    return logger;
}

int get_seed() {
    const char *seed_str = std::getenv("POPCORN_SEED");
    int seed = 42;
    if (seed_str) {
        seed = std::stoi(seed_str);
    }
    unsetenv("POPCORN_SEED");
    return seed;
}

std::vector<TestCase> get_test_cases(const std::string& tests_file_name) {
    std::unique_ptr<std::FILE, Closer> test_case_file;
    test_case_file.reset(::fopen(tests_file_name.c_str(), "r"));

    if(!test_case_file) {
        std::error_code ec(errno, std::system_category());
        std::cerr << "Could not open test file`" << tests_file_name << "`: " << ec.message() << std::endl;
        std::exit(ExitCodes::EXIT_PIPE_FAIL);
    }

    std::vector<TestCase> tests;
    while(true) {
        std::string line = read_line(test_case_file.get());
        tests.push_back(parse_test_case(line));

        // have we reached eof
        int peek = std::getc(test_case_file.get());
        if(peek != EOF) {
            std::ungetc(peek, test_case_file.get());
        } else {
            return tests;
        }
    }
}

template<class R, class... Args, std::size_t... Indices>
R call_generate_input_helper(const TestCase& tc, R(*func)(Args...), const std::index_sequence<Indices...>&) {
    return func(std::get<Indices>(tc.params)...);
}

template<class R, class... Args>
R call_generate_input(const TestCase& tc, R(*func)(Args...)) {
    return call_generate_input_helper(tc, func, std::make_index_sequence<sizeof...(Args)>{});
}

void warm_up(const TestCase& test) {
    using std::chrono::milliseconds;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;

    {
        auto warmup_data = call_generate_input(test, &generate_input);
        // warm up for at least 200 milliseconds
        auto start = high_resolution_clock::now();
        while(duration_cast<milliseconds>(high_resolution_clock::now() - start).count() < 200) {
            // discard result; this is just warmup, we don't care what it returns
            (void)custom_kernel(warmup_data);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}

struct BenchmarkStats {
    int runs;
    double mean;
    double std;
    double err;
    double best;
    double worst;
};

BenchmarkStats calculate_stats(const std::vector<std::int64_t>& durations) {
    int runs = (int)durations.size();
    // calculate duration statistics
    std::int64_t total_duration = std::accumulate(durations.begin(), durations.end(), (std::int64_t)0);
    std::int64_t best = *std::min_element(durations.begin(), durations.end());
    std::int64_t worst = *std::max_element(durations.begin(), durations.end());
    double average_duration = (double)total_duration / runs;

    double variance = 0.0;
    for(auto d : durations) {
        variance += std::pow((double)d - average_duration, 2);
    }

    // sample standard deviation with Bessel's correction
    double standard_deviation = std::sqrt(variance / (runs - 1));
    // standard error of the mean
    double standard_error = standard_deviation / std::sqrt(runs);

    return {runs, average_duration, standard_deviation, standard_error, (double)best, (double)worst};
}

BenchmarkStats benchmark(const TestCase& test) {
    std::vector<std::int64_t> durations;
    durations.reserve(100);

    // generate input data once
    auto data = call_generate_input(test, &generate_input);

    // now, do multiple timing runs without further correctness testing
    // there is an upper bound of 100 runs, and a lower bound of 3 runs;
    // otherwise, we repeat until we either measure at least 10 full seconds,
    // or the relative error of the mean is below 1%.

    for(int i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        // move data into custom_kernel, so that if custom_kernel takes large std::vectors or similar by value,
        // we're not measuring the copy overhead.
        auto submission_output = custom_kernel(std::move(data));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        durations.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        if(i > 1) {
            auto stats = calculate_stats(durations);
            // if we have enough data for an error < 1%
            // or if the total running time exceeds 10 seconds
            if((stats.err / stats.mean < 0.01) || (stats.mean * stats.runs > 10e9)) {
                break;
            }
        }
    }

    return calculate_stats(durations);
}


BenchmarkStats measure_for_leaderboard(PopcornOutput& logger, TestCase benchmark, int seed) {
    std::vector<std::int64_t> durations;
    durations.reserve(200);
    for (int i = 0; i < 200; ++i) {
        // TODO manipulate test case so that we use a different seed every time
        auto data = call_generate_input(benchmark, &generate_input);
        auto copy = data;
        auto start = std::chrono::high_resolution_clock::now();
        // move data into custom_kernel, so that if custom_kernel takes large std::vectors or similar by value,
        // we're not measuring the copy overhead.
        auto submission_output = custom_kernel(std::move(data));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        TestReporter reporter;
        check_implementation(reporter, copy, submission_output);

        durations.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        if(!reporter.has_passed()) {
            logger.log("check", "fail");
            logger.log("test.0.status", "fail");
            logger.log("test.0.error", reporter.message().c_str());
            exit(ExitCodes::EXIT_TEST_FAIL);
        }

        if (i > 1) {
            auto stats = calculate_stats(durations);
            // if we have enough data for an error < 1%
            // or if the total running time exceeds 30 seconds
            if ((stats.err / stats.mean < 0.01) || (stats.mean * stats.runs > 30e9)) {
                break;
            }
        }
    }

    logger.log("check", "pass");
    return calculate_stats(durations);
}

} // namespace

int main(int argc, const char* argv[]) {
    // setup
    PopcornOutput logger = open_logger();
    int seed = get_seed();

    if(argc < 3) {
        return ExitCodes::USAGE_ERROR;
    }

    std::string mode = argv[1];

    std::vector<TestCase> tests = get_test_cases(argv[2]);

    if(mode == "test" || mode == "benchmark") {
        bool pass = true;
        for (int i = 0; i < tests.size(); ++i) {
            auto& tc = tests.at(i);
            logger.log("test." + std::to_string(i) + ".spec", tc.spec.c_str());
            auto data = call_generate_input(tc, &generate_input);
            auto copy = data;
            auto submission_output = custom_kernel(std::move(data));

            TestReporter reporter;
            check_implementation(reporter, copy, submission_output);

            // log test status
            if (!reporter.has_passed()) {
                logger.log("test." + std::to_string(i) + ".status", "fail");
                logger.log("test." + std::to_string(i) + ".error", reporter.message().c_str());
                pass = false;
                // benchmark: stop at first failure; test: continue
                if(mode == "benchmark") {
                    break;
                }
            } else {
                logger.log("test." + std::to_string(i) + ".status", "pass");
            }
        }

        if(pass) {
            logger.log("check", "pass");
        } else {
            logger.log("check", "fail");
            return ExitCodes::EXIT_TEST_FAIL;
        }

        if(mode == "test")
            return EXIT_SUCCESS;
    }

    if (mode == "benchmark") {
        warm_up(tests.front());
        for (int i = 0; i < tests.size(); ++i) {
            const TestCase& tc = tests.at(i);
            auto result = benchmark(tc);
            logger.log("duration." + std::to_string(i) + ".spec", tc.spec.c_str());
            logger.log("duration." + std::to_string(i) + ".runs", result.runs);
            logger.log("duration." + std::to_string(i) + ".mean", result.mean);
            logger.log("duration." + std::to_string(i) + ".std", result.std);
            logger.log("duration." + std::to_string(i) + ".err", result.err);
            logger.log("duration." + std::to_string(i) + ".best", result.best);
            logger.log("duration." + std::to_string(i) + ".worst", result.worst);
        }
        return EXIT_SUCCESS;
    }

    if (mode == "leaderboard" ) {
        warm_up(tests.front());
        auto result = measure_for_leaderboard(logger, tests.back(), seed);
        logger.log("duration.spec", tests.back().spec.c_str());
        logger.log("duration.runs", result.runs);
        logger.log("duration.mean", result.mean);
        logger.log("duration.std", result.std);
        logger.log("duration.err", result.err);
        logger.log("duration.best", result.best);
        logger.log("duration.worst", result.worst);
    } else {
        std::cerr << "Unknown evaluation mode '" << mode << "'" << std::endl;
        return ExitCodes::USAGE_ERROR;
    }

    return EXIT_SUCCESS;
}
