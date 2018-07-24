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

  dim3_t _thread_sizes;  //!< The number of threads in each dim of a block.
  dim3_t _block_sizes;   //!< The number of blocks in each dim.

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
    constexpr auto dim         = Dimension<Value>();
    const     auto flux_solver = flux_solver_t(mat, dtdh);
              auto patch       = make_patch_iterator(in, dispatch_tag);

    // Shift the iterators to offset the padding, then set the patch data:
    shift_iterators(in, out, patch, dim);
    *patch = *in;
    __syncthreads();

    loader_t::load_boundary(in, patch, dim, setter);
    //load_boundary(in_iter, patch_iter, dim, setter);

      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
    *out = *patch + dtdh * flux_solver.flux_delta(patch, dim);
  }

 private:
  /// Offsets the global iterator in the given dimension.
  /// \param[in] it       The iterator to offset.
  /// \tparam    Iterator The type of te iterator.
  /// \tparam    Value    The value which defines the dimension.
  template <typename I1, typename I2, std::size_t Value>
  fluidity_host_device static auto
  shift_iterators(I1&& in, I1&& out, I2&& patch, Dimension<Value>)
  {
    constexpr auto dim = Dimension<Value>{};
    in.shift(flattened_id(dim), dim);
    out.shift(flattened_id(dim), dim);
    patch.shift(thread_id(dim) + padding, dim);
  }

  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename It>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_1d_t)
  {
    using state_t        = std::decay_t<decltype(*(it))>;
    using dim_info_t     = DimInfoCt<threads_per_block_1d_x>;
    using padding_info_t = PaddingInfo<dimx_t::value, padding>;
    return make_multidim_iterator<state_t, dim_info_t, padding_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename It>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_2d_t)
  {
    using state_t        = std::decay_t<decltype(*(it))>;
    using padding_info_t = PaddingInfo<dimy_t::value, padding>;
    using dim_info_t     = DimInfoCt<threads_per_block_2d_x,    
                                     threads_per_block_2d_y>;
    return make_multidim_iterator<state_t, dim_info_t, padding_info_t>();
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

  /// Solve function which invokes the split solver to use the \p in input data
  /// to compute the new state data and write the new data to \p out.
  /// \param[in] in_iter  The input multi dimensional iterator over state data.
  /// \param[in] out_iter The output multi dimensional iterator over state data
  ///                     to write the results to. 
  /// \param[in] material The material for the system.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Material The type of the material for the system.
  /// \tparam    T        The data type used for the simulation.
/*
  template <typename Iterator, typename Material, typename T>
  fluidity_device_only void solve(Iterator&&            in_iter ,
                                  Iterator&&            out_iter,
                                  Material              material,
                                  T                     dtdh    ,
                                  const BoundarySetter& setter  ) const
  {
    const auto flux_solver = flux_solver_t(material, dtdh);
    auto       patch_iter  = make_patch_iterator(in_iter, dispatch_tag);

    // Shift the iterators to offset the padding, then set the patch data:
    unrolled_for<num_dimensions>([&] (auto i)
    {
      shift_iterators(in_iter, out_iter, patch_iter, Dimension<i>{});
    });
    *patch_iter = *in_iter;
    __syncthreads();

    //unrolled_for<num_dimensions>([&] (auto i)
    //{
      // Defined for explicit conctexpr:
      //constexpr auto dim = Dimension<i>{};
      constexpr auto dim = Dimension<0>{};

      // Shift the iterators so that the correct cell is operated on:
      //shift_iterators(in_iter, out_iter, patch_iter, dim);

      // Load the global data into the shared data:
      //   *patch_iter = *in_iter;
      //__syncthreads();

      // Load in the data at the global and patch boundaries.
      // For dimensions != x, we use the shared memory to load the bounday:
      //loader_t::load_boundary(
      //  dim = dim_x ? in_iter : patch_iter, patch_iter, dim, setter);

      load_boundary(in_iter, patch_iter, dim, setter);

      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
      //   *out_iter = *patch_iter + dtdh * flux_solver.flux_delta(patch_iter, dim);
      *patch_iter = *patch_iter + dtdh * flux_solver.flux_delta(patch_iter, dim);
      __syncthreads();
 //   });
    *out_iter = *patch_iter;
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
  fluidity_device_only static void solve_impl(It&&             in    ,
                                              It&&             out   ,
                                              Mat&&            mat   ,
                                              T                dtdh  ,
                                              setter_ref_t     setter,
                                              Dimension<Value>       )
  {
    constexpr auto dim         = Dimension<Value>();
    const     auto flux_solver = flux_solver_t(mat, dtdh);
              auto patch_iter  = make_patch_iterator(in, dispatch_tag);

    // Shift the iterators to offset the padding, then set the patch data:
    shift_iterators(in_iter, out_iter, patch_iter, dim);
    *patch_iter = *in_iter;
    __syncthreads();

    loader_t::load_boundary(global_iter, patch_iter, dim, setter);
    //load_boundary(in_iter, patch_iter, dim, setter);

      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
    *out_iter = *patch_iter + dtdh * flux_solver.flux_delta(patch_iter, dim);
  }

 private:
  /// Defines the type to enable loading for the x dimension.
  /// \tparam V The value which defines the dimension.
  template <std::size_t V>
  using dim_x_enable_t = std::enable_if_t<V == 0, int>;

  /// Defines the type to enable loading other dimensions.
  /// \tparam V The value which defines the dimension.
  template <std::size_t V>
  using dim_not_x_enable_t = std::enable_if_t<V != 0, int>;



  /// Loads the boundary data for the x dimension. For the x dimension, data
  /// which is not at the boundary is loaded into the \p patch_iter using the
  /// data from the \p global_iter.
  /// \param[in] global_iter An iterator over global data.
  /// \param[in] patch_iter  An iterator over patch data.
  /// \param[in] setter      The setter function to set boundary data.
  /// \tparam    I1          The type of the global iterator.
  /// \tparam    I2          The type of the patch iterator.
  /// \tparam    V           The value which defines the dimension.
  template <typename I1, typename I2, std::size_t V, dim_x_enable_t<V> = 0>
  fluidity_device_only void
  load_boundary(I1&&                  global_iter,
                I2&&                  patch_iter ,
                Dimension<V>                     ,
                const BoundarySetter& setter     ) const
  {
    constexpr auto dim = Dimension<V>();
    loader_t::load_boundary(global_iter, patch_iter, dim, setter);
  }

  /// Loads the boundary data for the all dimensions other than the x dimension.
  /// After performing sweeps in the previous dimensions the results are stored
  /// in the patch (shared memory), so rather than using the global iterator for
  /// to load in boundary and other data, the patch data is used for both.
  /// \param[in] patch_iter  An iterator over patch data.
  /// \param[in] setter      The setter function to set boundary data.
  /// \tparam    I1          The type of the global iterator.
  /// \tparam    I2          The type of the patch iterator.
  /// \tparam    V           The value which defines the dimension.
  template <typename I1, typename I2, std::size_t V, dim_not_x_enable_t<V> = 0>
  fluidity_device_only void
  load_boundary(I1&&                            ,
                I2&&                  patch_iter,
                Dimension<V>                    ,
                const BoundarySetter& setter    ) const
  {
    constexpr auto dim = Dimension<V>();
    loader_t::load_boundary(patch_iter, patch_iter, dim, setter);
  }

  /// Offsets the global iterator in the given dimension.
  /// \param[in] it       The iterator to offset.
  /// \tparam    Iterator The type of te iterator.
  /// \tparam    Value    The value which defines the dimension.
  template <typename DIterator, typename PIterator, std::size_t Value>
  fluidity_host_device auto shift_iterators(DIterator&& in,
                                            DIterator&& out,
                                            PIterator&& pit,
                                            Dimension<Value>) const
  {
    constexpr auto dim = Dimension<Value>{};
    in.shift(flattened_id(dim), dim);
    out.shift(flattened_id(dim), dim);
    pit.shift(thread_id(dim) + padding, dim);
  }
  */
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_HPP