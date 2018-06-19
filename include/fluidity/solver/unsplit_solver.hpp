//==--- fluidity/solver/unsplit_solver.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unsplit_solver.hpp
/// \brief This file defines implementations of an unsplit solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP
#define FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP

#include "solver_utilities.hpp"

namespace fluid  {
namespace solver {

/// The UnsplitSolver class defines an implementatio of a solver which updates
/// states using a dimensionally unsplit method. It can be specialized for
/// systems of different dimension. This implementation is defaulted for the 1D
/// case.
/// \tparam Traits     The componenets used by the solver.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename FluxSolver, typename Loader, std::size_t Dimensions = 1>
struct UnsplitSolver {
 private:
  /// Defines the type of the loader for the data.
  using loader_t = std::decay_t<Loader>;

  /// Defines the number of dimensions to solve over.
  static constexpr std::size_t num_dimensions = 1;
  /// Defines the amount of padding in the data loader.
  static constexpr std::size_t padding        = loader_t::padding;

 public:
  /// Solve function which invokes the solver.
  /// \param[in] data     The iterator which points to the start of the global
  ///            state data. If the iterator does not have 1 dimension then a
  ///            compile time error is generated.
  /// \param[in] flux     An iterator which points to the flux to update. 
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Flux     The type of the flux iterator.
  template <typename It, typename T>
  fluidity_device_only void solve(It&& in, It&& out, T dtdh) const
  {
    static_assert(std::decay_t<It>::num_dimensions() == num_dimensions,
                  "Dimensions of iterator do not match solver specialization");

  }
};

}} // namespace fluid::solver


#endif // FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP