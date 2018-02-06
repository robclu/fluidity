//==--- fluidity/test/cuda_test.cu-------------------------- -*- C++ -*- ---==//
//            
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cuda_test.cu
/// \brief This file defines a simple test case to check that cuda works.
//
//==------------------------------------------------------------------------==//

#include <fluidity/utility/debug.hpp>
#include <stdlib.h>
#include <stdio.h>

__global__ void invoke_kernel(float sum, int iterations) {
  for (auto i = 0; i < iterations; ++i)
    sum += 1.0f;

  auto thread = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Thread %02u, value %04.4f\n", thread, sum);
}

int main(int argc, char** argv) {
  int threads = 16, iterations = 1;
  float sum = 0.0f;

  if (argc > 1)
    threads = atoi(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);

  invoke_kernel<<<1, threads>>>(sum, iterations);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}
