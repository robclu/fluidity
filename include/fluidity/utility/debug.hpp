//==--- fluidity/utility/debug.hpp ------------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  debug.hpp
/// \brief This file defines debugging functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_DEBUG_HPP
#define FLUIDITY_UTILITY_DEBUG_HPP

#include <cassert>
#include <exception>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef MAX_UNROLL_DEPTH
  #define MAX_UNROLL_DEPTH 0
#endif  // MAX_UNROLL_DEPTH

namespace fluid {

/// Defines the maximum loop size where compile time unrolling may be performed
/// instead of a traditional loop.
static constexpr std::size_t max_unroll_depth = MAX_UNROLL_DEPTH;

namespace util {

/// Checks if a cuda error code was a success, and if not, then it prints
/// the error message.
/// \param[in] errCode The cuda error code.
/// \param[in] file    The file where the error was detected.
/// \param[in] line    The line in the file where the error was detected.
inline void check_cuda_error(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess)
  {
    printf("\nCuda Error : %s\nFile       : %s\nLine       :  %i\n\n",
      cudaGetErrorString(code), file, line);
    std::terminate();
  }
}

} // namespace util
} // namespace fluid

#if defined(NDEBUG)

/// Defines a macro for checking a cuda error in release mode. This does not
/// do anything so that there is no performance cost in release mode.
#define fluidity_check_cuda_result(result) (result)

#else

/// Defines a macro to check the result of cuda calls in debug mode.
#define fluidity_check_cuda_result(result)                                  \
  ::fluid::util::check_cuda_error((result), __FILE__, __LINE__)

#endif // NDEBUG

#endif // FLUIDITY_UTILITY_DEBUG_HPP
