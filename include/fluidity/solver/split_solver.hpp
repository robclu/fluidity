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
#include <fluidity/dimension/dimension.hpp>
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
  using flux_solver_t    = std::decay_t<FluxSolver>;
  /// Defines the type of the loader for the data.
  using loader_t         = std::decay_t<Loader>;

  /// Defines the number of dimensions to solve over.
  static constexpr auto num_dimensions  = Dimensions;
  /// Defines the amount of padding in the data loader.
  static constexpr auto padding         = loader_t::padding;
  /// Defines the dispatch tag for dimension overloading.
  static constexpr auto dispatch_tag    = dim_dispatch_tag<num_dimensions>;

 public:
  /// Solve function which invokes the split solver to use the \p in input data
  /// to compute the new state data and write the new data to \p out.
  /// \param[in] in_iter  The input multi dimensional iterator over state data.
  /// \param[in] out_iter The output multi dimensional iterator over state data
  ///                     to write the results to. 
  /// \param[in] material The material for the system.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Material The type of the material for the system.
  /// \tparam    T        The data type used for the simulation.
  template <typename Iterator, typename Material, typename T>
  fluidity_device_only void solve(Iterator&&            in_iter ,
                                  Iterator&&            out_iter,
                                  Material              material,
                                  T                     dtdh    ,
                                  const BoundarySetter& setter  ) const
  {
    const auto flux_solver = flux_solver_t(material, dtdh);
    auto       patch_iter  = make_patch_iterator(in_iter, dispatch_tag);

    unrolled_for<num_dimensions>([&] (auto i)
    {
      // Defined for explicit conctexpr:
      constexpr auto dim = Dimension<i>{};

      // Shift the iterators so that the correct cell is operated on:
      shift_iterators(in_iter, out_iter, patch_iter, dim);

      // Load the global data into the shared data:
      *patch_iter = *in_iter;
      __syncthreads();

      // Load in the data at the global and patch boundaries:
      loader_t::load_boundary(in_iter, patch_iter, dim, setter);

      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
      *out_iter = *patch_iter + dtdh * flux_solver.flux_delta(patch_iter, dim);
    });
  }

 private:
  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto
  make_patch_iterator(Iterator&& it, dispatch_tag_1d_t) const
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<threads_per_block_1d_x>;
    return make_multidim_iterator<state_t, dim_info_t, padding>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto
  make_patch_iterator(Iterator&& it, dispatch_tag_2d_t) const
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<threads_per_block_2d_x,    
                                 threads_per_block_2d_y>;
    return make_multidim_iterator<state_t, dim_info_t, padding>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto
  make_patch_iterator(Iterator&& it, dispatch_tag_3d_t) const
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<threads_per_block_3d_x,    
                                 threads_per_block_3d_y,
                                 threads_per_block_3d_z>;
    return make_multidim_iterator<state_t, dim_info_t, padding>();
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
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_HPP