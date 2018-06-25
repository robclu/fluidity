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
/// \tparam Traits     The components used by the solver.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename FluxSolver, typename Loader, std::size_t Dimensions = 1>
struct SplitSolver {
 private:
  /// Defines the type of the flux solver.
  using flux_solver_t    = std::decay_t<FluxSolver>;
  /// Defines the type of the loader for the data.
  using loader_t         = std::decay_t<Loader>;

  /// Defines the number of dimensions to solve over.
  static constexpr std::size_t num_dimensions = Dimensions;
  /// Defines the amount of padding in the data loader.
  static constexpr std::size_t padding        = loader_t::padding;

 public:
  /// Solve function which invokes the split solver to use the \p in input data
  /// to compute the new state data and write the new data to \p out.
  /// \param[in] in       The input state data.
  /// \param[in] out      The output state data to write the results to.
  /// \param[in] material The material for the system.
  ///
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator, typename Material, typename T>
  fluidity_device_only void solve(Iterator&&            in      ,
                                  Iterator&&            out     ,
                                  Material              material,
                                  T                     dtdh    ,
                                  const BoundarySetter& setter  ) const
  {
    const auto flux_solver = flux_solver_t(material, dtdh);
    
    auto global_iter = get_global_iterator(in);
    auto patch_iter  = get_patch_iterator(in);

    // Load the global data into the shared data:
    *patch_iter = *global_iter;
    __syncthreads();

    // Load in the data at the global and patch boundaries:
    loader_t::load_boundary(global_iter, patch_iter, dim_x, setter);

    // Update states as : U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
    *get_global_iterator(out) = 
      *patch_iter + dtdh * flux_solver.flux_delta(patch_iter, dim_x);
  }

 private:
  /// Returns a global multi dimensional iterator which is shifted to the global
  /// thread index in the x-dimension.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto get_global_iterator(Iterator&& it) const
  {
    using dim_info_t = DimInfo<num_dimensions>;
    auto output_it   = make_multidim_iterator(it.get_ptr()              ,
                                              dim_info_t{it.size(dim_x)});
    return output_it.offset(flattened_id(dim_x), dim_x);
  }

  /// Returns a shared multi dimensional iterator which is offset by the amount
  /// of padding and the local thread index, so the iterator points to the
  /// appropriate data for the thread to operate on.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto get_patch_iterator(Iterator&& it) const
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<default_threads_per_block>;

    auto output_it = make_multidim_iterator<state_t, dim_info_t, padding>();
    return output_it.offset(thread_id(dim_x) + padding, dim_x);
  }
};

}} // namespace fluid::solver


#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_HPP