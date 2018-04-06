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

/// Used to define the type of compile time dimension information for a patch
/// for the update implementation. It can be specialized based on the number of
/// dimensions.
/// \tparam Dims  The number of dimensions.
template <std::size_t Dims>
struct MakePatchInfo;

/// Specialization for a single dimension.
template <>
struct MakePatchInfo<1> { 
  /// Defines the type of compile time dimension information.
  using type = DimInfoCt<default_threads_per_block>;
};

/// Specialization for two dimensions.
template <>
struct MakePatchInfo<2> { 
  /// Defines the type of compile time dimension information.
  using type = 
    DimInfoCt<
      (default_threads_per_block >> 2)
    , (default_threads_per_block >> 2)
    >;
};

/// Specialization for two dimensions.
template <>
struct MakePatchInfo<3> { 
  /// Defines the type of compile time dimension information.
  using type = 
    DimInfoCt<
      (default_threads_per_block >> 3)
    , (default_threads_per_block >> 3)
    , (default_threads_per_block >> 3)
    >;
};

template < typename    Iterator,
         , std::size_t Padding ,
         , std::enable_if_t<Iterator::dimensions == 1, int> = 0
         >
void shift_for_padding(Iterator& iter)
{
  iter.shift(Padding, dim_x);
}

template < typename    Iterator,
         , std::size_t Padding ,
         , std::enable_if_t<Iterator::dimensions == 2, int> = 0
         >
void shift_for_padding(Iterator& iter)
{
  iter.shift(Padding, dim_x).shift(Padding, dim_y);
}

template < typename    Iterator,
         , std::size_t Padding ,
         , std::enable_if_t<Iterator::dimensions == 3, int> = 0
         >
void shift_for_padding(Iterator& iter)
{
  iter.shift(Padding, dim_x).shift(Padding, dim_y).shift(Padding, dim_z);
}

} // namespace detail

/// Alias for the type of patch information for updating.
template <std::size_t Dims>
using patch_info_t = typename detail::MakePatchInfo<Dims>::type;

template <typename Iterator, typename Loader>
fluidity_global void update_impl(Iterator begin, Iterator end)
{
  using patch_info_t = patch_info_t<Iterator::dimensions>;
  using state_t      = std::decay_t<decltype(*begin)>;

  auto global_iter = make_multidim_iterator<state_t, patch_info_t>(&(*begin));
  auto patch_iter  = make_multidim_iterator<state_t, patch_info_t>();

  detail::shift_for_padding<Loader::padding>(global_iter);
  detail::shift_for_padding<Loader::padding>(patch_iter);

  // Move the iterators passed the padding:
  *patch_iter = *global_iter;

  unrolled_for<dim_info_t::num_dimensions()>([&] (auto dim_value)
  {
    constexpr auto dim      = Dimension<dim_value>();
    constexpr auto dim_size = dim_info_t::size(dim);

    // Move the iterators:

    loader.load_internal(patch_iter, dim_size, dim);
    loader.load_boundary(global_iter, patch_iter, grid_size(dim), dim);
    __syncthreads();
  });
}

template <typename Iterator, typename Loader>
void update(Iterator begin, Iterator end, Loader loader)
{
#if defined(__CUDACC__)
  constexpr auto padding     = Loader::padding;
  constexpr auto max_threads = default_threads_per_block;

  auto threads_per_block = 
    detail::get_threads_per_block<padding>(Dimension<Iterator::dimensions>{});

  dim3 threads_per_block(elements < max_threads ? elements : max_threads);
  dim3 num_blocks(std::max(elements / threads_per_block.x,
                           static_cast<unsigned int>(1)));

  update_impl<<num_blocks, threads_per_block>>>(begin, end);
  fluidity_cuda_check_result(cudaDeviceSynchronize()); 
#endif // __CUDACC__
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH