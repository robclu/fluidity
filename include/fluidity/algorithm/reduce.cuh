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
/// \param[in] begin     The first iterator to fill with the \p value.
/// \param[im] size      The size of the data to reduce.
/// \param[in] pred      The predicate to use for the reduction.
/// \param[in] args      Additional arguments for the predicate.
/// \tparam    Iterator  The type of the iterator.
/// \tparam    BlockInfo Information for the block.
/// \tparam    Pred      The type of the predicate.
/// \tparam    Args      The type of the arguments for the predicate.
template < typename    Iterator
         , typename    BlockInfo
         , typename    Pred
         , typename... Args
         >
fluidity_global void reduce_impl(Iterator    begin  ,
                                 Iterator    results,
                                 std::size_t size   ,
                                 Pred        pred   ,
                                 Args...     args   )
{
  const auto index = flattened_id(dim_x);
  if (index < offset)
  {
    // Create a shared memory multidimensional iterator:
    auto iter = make_multidim_iterator<typename Iterator::value_t, BlockInfo>();

    // Load the data into shared memory:
    *iter = *begin[index]
    __syncthreads();

    const auto block_start = flattened_block_id(dim_x) * block_size(dim_x);
    const auto leftover    = (size << 1) - block_start;

    // Number of elements to reduce in this block is the smaller of the block
    // size and the extra elements for non-power-of-2 block sizes ...
    auto size = std::min(block_size(), leftover);
    while (index < (size >> 1))
    {
      size = (size >> 1) + (size & 1);
      pred(*iter, *(iter.offset(size)), args...);
      __syncthreads();
    }

    if (index == 0)
    {
      *(results[flattened_block_id(dim_x)]) = *iter;
    }
  }
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
  const int      elements    = end - begin;
  constexpr auto max_threads = 256;

  dim3 threads_per_block(elements < max_threads ? elements : max_threads);
  dim3 num_blocks(std::max(elements / threads_per_block.x,
                           static_cast<unsigned int>(1)));

  using dim_info_t     = DimInfoCx<max_threads, 1, 1>;
  using value_t        = decay_t<Iterator>::value_t;
  using host_results_t = HostTensor<decay_t<value_t, 1>>;
  using dev_results_t  = DeviceTensor<decay_t<value_t, 1>>;

  dev_results_t dev_results(num_blocks.x);
  reduce_impl<dim_info_t><<<num_blocks, threads_per_block>>>(
    begin              ,
    dev_results.begin(),
    elements           ,
    pred               ,
    args...
  );
  fluidity_check_cuda_result(cudaDeviceSynchronize());

  auto host_results = host_results_t(dev_results);
  value_t result = *begin;
  for (const auto& e : host_results)
  {
    pred(result, e);
  }
  return result;
#endif // __CUDACC__ 
}

}}} // namespace fluid::detail::cuda


#endif // FLUIDITY_ALGORITHM_REDUCE_CUH