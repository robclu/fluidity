//==--- fluidity/utility/portability.hpp ------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  portability.hpp
/// \brief This file defines utilities for portability.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_PORTABILITY_HPP
#define FLUIDITY_UTILITY_PORTABILITY_HPP

#include <fluidity/execution/execution_policy.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

/// Definitions for host, device, and host device functions, if CUDA is
/// supported.
#if defined(__CUDACC__) || defined(__clang__)
  #define fluidity_host_only   __host__
  #define fluidity_device_only __device__
  #define fluidity_host_device __host__ __device__ 
#else
  #define fluidity_host_only
  #define fluidity_device_only
  #define fluidity_host_device
#endif

namespace fluid {
namespace exec  {

// If the compilation system has cuda functionality then set the default
// execution policy to use the GPU.
#if defined(FLUIDITY_CUDA_AVAILABLE)

/// Defines the default type of execution to use.
using default_type = gpu_type;

/// If the compilation system has cuda functionality then set the default
/// execution policy to use the GPU.
static constexpr auto default_policy = gpu_policy;

#else

/// Defines the default type of execution to use.
using default_type = cpu_type;

/// If the compilation system has no cuda functionality then set the default
/// execution policy to use the CPU.
static constexpr auto default_policy = cpu_policy;

#endif // FLUIDITY_CUDA_AVAILABLE

}} // namespace fluid::exec

#endif // FLUIDITY_UTILITY_PORTABILITY_HPP