#ifndef __REFERENCE_CUH__
#define __REFERENCE_CUH__

#include <tuple>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <array>
#include <iostream>

#define N_SIZES 10
const int Ns[N_SIZES] = {128,  256,  512,   1024,  2048,
                         4096, 8192, 16384, 32768, 65536};

using input_t = std::array<std::vector<float>, N_SIZES>;
using output_t = input_t;

inline input_t generate_input() {
  input_t data;

  for (int i = 0; i < N_SIZES; ++i) {
    data[i].resize(Ns[i]);
    for (int j = 0; j < Ns[i]; ++j) {
      data[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  return data;
}

// The identity kernel
inline output_t ref_kernel(input_t data) {
  return (output_t) data;
}

inline bool check_implementation(const input_t& data, const output_t& out, float epsilon = 1e-5) {
  output_t ref = ref_kernel(data);

  for (int i = 0; i < N_SIZES; ++i) {
    auto ref_ptr = ref[i];
    auto out_ptr = out[i];

    if(out[i].size() != Ns[i]) {
        std::cerr <<  "SIZE MISMATCH at " << i << ": " << Ns[i] << " " << out[i].size() << std::endl;
        return false;
    }

    for (int j = 0; j < Ns[i]; ++j) {
      if (std::fabs(ref_ptr[j] - out_ptr[j]) > epsilon) {
        std::cerr <<  "ERROR AT " << i << ", "<< j << ": " << ref_ptr[j] << " " << out_ptr[j] << std::endl;
        return false;
      }
    }
  }

  return true;
}

#endif
