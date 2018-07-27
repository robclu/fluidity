//==--- fluidity/solver/split_solver.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  split_solver.hpp
/// \brief This file defines implementations of a split solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_SPLIT_SOLVER_HPP
#define FLUIDITY_SOLVER_SPLIT_SOLVER_HPP

#include "solver_utilities.hpp"
#include "split_solver.cuh"
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/utility/cuda.hpp>

namespace fluid  {
namespace solver {

/// The SplitSolver class defines an implementation of a solver which updates
/// states using a dimensionally split method. It can be specialized for
/// systems of different dimension. This implementation is defaulted for the 1D
/// case.
/// \tparam FluxSolver The method to use for solving the face fluxes.
/// \tparam Loader     The implementation for boundary loading.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename FluxSolver, typename Loader, std::size_t Dimensions = 1>
struct SplitSolver {
 private:
  /// Defines the type of the flux solver.
  using flux_solver_t = std::decay_t<FluxSolver>;
  /// Defines the type of the loader for the data.
  using loader_t      = std::decay_t<Loader>;
  /// Defines the type of the boundary setter.
  using setter_t      = BoundarySetter;
  /// Defines a reference type to the boundary setter.
  using setter_ref_t  = const BoundarySetter&;

  /// Defines the number of dimensions to solve over.
  static constexpr auto num_dimensions  = Dimensions;
  /// Defines the amount of padding in the data loader.
  static constexpr auto padding         = loader_t::padding;
  /// Defines the dispatch tag for dimension overloading.
  static constexpr auto dispatch_tag    = dim_dispatch_tag<num_dimensions>;

#if defined(FLUIDITY_CUDA_AVAILABLE)
  dim3 _thread_sizes;  //!< The number of threads in each dim of a block.
  dim3 _block_sizes;   //!< The number of blocks in each dim.
#else
  Dim3   _thread_sizes;
  Dim3   _block_sizes;
#endif // FLUIDITY_CUDA_AVAILBLE

 public:
  /// Creates the split solver, initializing the number of threads and blocks
  /// based on the dimensionality of the iterator.
  /// \param[in] it The iterator to use for solving.
  /// \tparam    It The type of the iteraotr.
  template <typename It>
  SplitSolver(It&& it)
  : _thread_sizes{get_thread_sizes(it)}, 
    _block_sizes{get_block_sizes(it, _thread_sizes)} {}

  /// Updater function for updating the simulation. This overload is only
  /// enabled when the input and output iterators are for GPU execution.
  /// This simply forwards all the arguments onto the cuda implementation of the
  /// solving function.
  /// 
  /// \param[in] in    The input data to use to update.
  /// \param[in] out   The output data to write to after updating.
  /// \param[in] mat   The material for the system.
  /// \param[in] dtdh  Scaling factor for the update.
  /// \tparam    It    The type of the multi dimensional iterator.
  /// \tparam    Mat   The type of the material for the system.
  /// \tparam    T     The type of the scaling factor.
  template <typename It, typename Mat, typename T, exec::gpu_enable_t<It> = 0>
  void solve(It&& in, It&& out, Mat&& mat, T dtdh, setter_ref_t setter) const 
  {
    detail::cuda::solve_impl(*this                 ,
                             std::forward<It>(in)  ,
                             std::forward<It>(out) ,
                             std::forward<Mat>(mat),
                             dtdh                  ,
                             setter                );
  }

  /// Returns the number of threads per block for the solver.
  auto thread_sizes() const
  {
    return _thread_sizes;
  }

  /// Returns the dimension information for the blocks to solve.
  auto block_sizes() const
  {
    return _block_sizes;
  }

  /// Overlaod of the call operator to invoke a pass of solving on the input and
  /// output data iterators for a specific dimension which is defined by Value.
  /// The data from the \p in iterator is used to compute the update which is
  /// then written to the \p out iterator.
  /// \param[in] in     The input multi dimensional iterator over state data.
  /// \param[in] out    The output multi dimensional iterator over state data.
  /// \param[in] mat    The material for the system.
  /// \tparam    It     The type of the iterator.
  /// \tparam    Mat    The type of the material for the system.
  /// \tparam    T      The data type for the scaling factor.
  /// \tparam    Value  The value which defines the dimension for the pass.
  template <typename It, typename Mat, typename T, std::size_t Value>
  fluidity_device_only static void invoke(It&&             in    ,
                                          It&&             out   ,
                                          Mat&&            mat   ,
                                          T                dtdh  ,
                                          setter_ref_t     setter,
                                          Dimension<Value>       )
  {
    constexpr auto dim = Dimension<Value>();
    if (in_range(in))
    {
      const auto flux_solver = flux_solver_t(mat, dtdh);
            auto patch       = make_patch_iterator(in, dispatch_tag, dim);

      // Shift the iterators to offset the padding, then set the patch data:
      unrolled_for<num_dimensions>([&] (auto i) 
      {
        constexpr auto dim_offset = Dimension<i>();
        shift_iterators(in, out, patch, dim, dim_offset);
      });
      *patch = *in;

      loader_t::load_boundary(in, patch, dim, setter);
      __syncthreads();

      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
      *out = *patch + dtdh * flux_solver.flux_delta(patch, dim);
    }
  }

 private:
  /// Offsets the global iterator in the given dimension.
  /// \param[in] it       The iterator to offset.
  /// \tparam    Iterator The type of te iterator.
  /// \tparam    Value    The value which defines the dimension.
  template <typename I1, typename I2, std::size_t VS, std::size_t VO>
  fluidity_host_device static auto
  shift_iterators(I1&& in, I1&& out, I2&& patch, Dimension<VS>, Dimension<VO>)
  {
    constexpr auto dim_off = Dimension<VO>{};
    in.shift(flattened_id(dim_off), dim_off);
    out.shift(flattened_id(dim_off), dim_off);
    patch.shift(thread_id(dim_off) + (VS == VO ? padding : 0), dim_off);
  }

  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename It>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_1d_t)
  {
    using state_t        = std::decay_t<decltype(*(it))>;
    using dim_info_t     = DimInfoCt<threads_per_block_1d_x + (padding << 1)>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is called for a 2D system when solving in the x direction.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename It>
  fluidity_device_only static auto
  make_patch_iterator(It&& it, tag_2d_t, dimx_t)
  {
    using state_t        = std::decay_t<decltype(*(it))>;
    using dim_info_t     = DimInfoCt<threads_per_block_2d_x + (padding << 1),
                                     threads_per_block_2d_y>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is called for a 2D system when solving in the y direction.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename It>
  fluidity_device_only static auto
  make_patch_iterator(It&& it, tag_2d_t, dimy_t)
  {
    using state_t        = std::decay_t<decltype(*(it))>;
    using dim_info_t     = DimInfoCt<threads_per_block_2d_x,    
                                     threads_per_block_2d_y + (padding << 1)>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename It>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_3d_t)
  {
    using state_t        = std::decay_t<decltype(*(it))>;
    using padding_info_t = PaddingInfo<dimz_t::value, padding>;
    using dim_info_t     = DimInfoCt<threads_per_block_3d_x,    
                                     threads_per_block_3d_y,
                                     threads_per_block_3d_z>;
    return make_multidim_iterator<state_t, dim_info_t, padding_info_t>();
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_HPP