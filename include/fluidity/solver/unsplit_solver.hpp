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
template <typename FluxSolver, typename Loader, typename Dims = Num<1>>
struct UnsplitSolver {
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
  static constexpr auto num_dimensions = std::size_t{Dims()};
  /// Defines the amount of padding in the data loader.
  static constexpr auto padding        = loader_t::padding;

 public:
  template <typename It>
  UnsplitSolver(It&& it) {}

  /// Solve function which invokes the solver.
  /// \param[in] data     The iterator which points to the start of the global
  ///            state data. If the iterator does not have 1 dimension then a
  ///            compile time error is generated.
  /// \param[in] flux     An iterator which points to the flux to update. 
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Flux     The type of the flux iterator.
  template <typename It, typename Mat, typename T, exec::gpu_enable_t<It> = 0>
  void solve(It&& in, It&& out, Mat&& mat, T dtdh, setter_ref_t setter) const 
  {

  }
};

}} // namespace fluid::solver


#endif // FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP