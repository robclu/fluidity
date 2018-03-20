//==--- fluidity/algorithm/reduce.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce.cuh
/// \brief This file defines the implementation of the cuda version of reduction
///        functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_REDUCE_CUH
#define FLUIDITY_ALGORITHM_REDUCE_CUH

#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <cstddef>

namespace fluid  {
namespace detail {
namespace cuda   {

/// Kernel implementation which reduces a container where the first element it
/// pointed to by \p begin.
/// \param[in] begin    The first iterator to fill with the \p value.
/// \param[im] offset   The offset between \p begin and the next reduction
///            element.
/// \param[in] pred     The predicate to use for the reduction.
/// \param[in] args     Additional arguments for the predicate.
/// \tparam    Iterator The type of the iterator.
/// \tparam    Pred     The type of the predicate.
/// \tparam    Args     The type of the arguments for the predicate.
template <typename Iterator, typename Pred, typename... Args>
fluidity_global void reduce_impl(Iterator    begin ,
                                 std::size_t offset,
                                 Pred        pred  ,
                                 Args...     args  )
{
/*
  const auto index = flattened_id(dim_x);
  if (index < offset)
  {
    // Load the data into shared memory:
    *iter                   = *begin;
    *(iter[index + offset]) = *(begin[index + offset]);

    const auto leftover = (offset << 1)
                        - flattened_id_block(dim_x)
                        * block_size();
    auto size = std::min(block_size(), leftover);

    while (index < (size >> 1))
    {
      size = (size >> 1) + (size & 1);
      pred(*iter, *(iter[size]), args...);
      __syncthreads();
    }

    if (index == 0)
    {
      *begin = *iter;
    }
  }
*/
}

/// Wrapper function which invokes the cuda reduction kernel.
/// \param[in]  begin     An iterator to the beginning of the data.
/// \param[in]  end       An iterator to the end of the data.
/// \param[in]  pred      The predicate to apply to element pairs.
/// \param[in]  args      Additional arguments for the predicate.
/// \tparam     Iterator  The type of the iterator.
/// \tparam     Pred      The type of the predicate.
/// \tparam     Args      The type of any additional args for the predicate.
template <typename Iterator, typename Pred, typename... Args>
fluidity_host_device decltype(auto)
reduce(Iterator&& begin, Iterator&& end, Pred&& pred, Args&&... args)
{
#if defined(__CUDACC__)
  const int      elements    = (end - begin) / 2;
  constexpr auto max_threads = 256;

  dim3 threads_per_block(elements < max_threads ? elements : max_threads);
  dim3 num_blocks(std::max(elements / threads_per_block.x,
                           static_cast<unsigned int>(1)));

  reduce_impl<<<num_blocks, threads_per_block>>>(begin, value);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
#endif // __CUDACC__ 
}

}}} // namespace fluid::detail::cuda


#endif // FLUIDITY_ALGORITHM_REDUCE_CUH