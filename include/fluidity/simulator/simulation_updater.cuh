//==--- fluidity/simulator/simulation_updater.cuh ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_updater.cuh
/// \brief This file defines a class which updates a simulation using the GPU.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH
#define FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH

#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace sim    {
namespace detail {

/// Used to define the type of compile time dimension information for the
/// update implementation. It can be specialized based on the number of
/// dimensions.
/// \tparam Dims  The number of dimensions.
template <std::size_t Dims>
struct MakeDimInfoCt;

/// Specialization for a single dimension.
template <>
struct MakeDimInfoCt<1> { 
  /// Defines the type of compile time dimension information.
  using type = DimInfoCt<default_threads_per_block>;
};

/// Specialization for two dimensions.
template <>
struct MakeDimInfoCt<2> { 
  /// Defines the type of compile time dimension information.
  using type = 
    DimInfoCt<default_threads_per_block, default_threads_per_block>;
};

/// Specialization for two dimensions.
template <>
struct MakeDimInfoCt<3> { 
  /// Defines the type of compile time dimension information.
  using type = 
    DimInfoCt<
      default_threads_per_block
    , default_threads_per_block
    , default_threads_per_block
    >;
};

template <std::size_t Dims>
using get_dim_info_t = typename MakeDimInfoCt<Dims>::type;

} // namespace detail

template <typename Iterator>
fluidity_global void update_impl(Iterator begin)
{
  using dim_info_t = detail::get_dim_info_t<Iterator::dimensions>;
  using state_t    = std::decay_t<decltype(*begin)>;

  auto global_iter = make_multidim_iterator<state_t, dim_info_t>(*begin);
  auto patch_iter  = make_multidim_iterator<state_t, dim_info_t>();

  *patch_iter = *global_iter;
  unrolled_for<dim_info_t::num_dimensions()>([&] (auto dim_value)
  {
    constexpr auto dim      = Dimension<dim_value>();
    constexpr auto dim_size = dim_info_t::size(dim);

    loader.load_internal(patch_iter , dim_size, dim);
    loader.load_boundary(global_iter, patch_iter, dim_size, dim);
    __syncthreads();
  });
}

template <typename Iterator>
void update(Iterator begin, Iterator end)
{
#if defined(__CUDACC__)
  const int elements         = end - begin;
  constexpr auto max_threads = default_threads_per_block;

  dim3 threads_per_block(elements < max_threads ? elements : max_threads);
  dim3 num_blocks(std::max(elements / threads_per_block.x,
                           static_cast<unsigned int>(1)));

  update_impl<<num_blocks, threads_per_block>>><dim_info_t>(begin, elements);
  fluidity_cuda_check_result(cudaDeviceSynchronize()); 
#endif // __CUDACC__
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH