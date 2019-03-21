//==--- fluidity/utility/cuda.cuh -------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cuda.cuh
/// \brief This file defines cuda related utility kernels.
//
//==------------------------------------------------------------------------==//

#include "portability.hpp"
#include <fluidity/dimension/thread_index.hpp>

#ifndef FLUIDITY_UTILITY_CUDA_CUH
#define FLUIDITY_UTILITY_CUDA_CUH

namespace fluid {
namespace util  {
namespace cuda  {

/// Copies each data element from \p in to \p out.
/// \param[in]  in       A pointer to the input data.
/// \param[out] out      A pointer to the output data.
/// \param      elements The number of elements to set.
/// \tparam     Ptr      The type of the pointers.
template <typename Ptr>
fluidity_global void copy(const Ptr* in, Ptr* out, std::size_t elements)
{
  const auto idx = flattened_id(dim_x);
  if (idx < elements)
  {
    out[idx] = in[idx];
  }
}

}}} // namespace fluid::util::cuda

#endif // FLUIDITY_UTILITY_CUDA_CUH