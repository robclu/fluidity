//==--- fluidity/reconstruction/riemann_input.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  riemann_input.hpp
/// \brief This file defines a simple struct for the input states which can be
///        used to solve a Riemann problem.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_RECONSTRUCTION_RIEMANN_INPUT_HPP
#define FLUIDITY_RECONSTRUCTION_RIEMANN_INPUT_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace recon {

/// The RiemannInput struct defines a simple container for the input to a
/// Riemann Problem. It defines a left and a right state for the problem.
/// \tparam State The type of the states which must be used in the solution.
template <typename State>
struct RiemannInput {
  State left;   //!< The left input state for the Riemann Problem.
  State right;  //!< The right input state for the Riemann Problem.
};


/// Utility function which can make the input for a Riemann Problem.
template <typename State>
fluidity_host_device auto make_riemann_input(State&& l, State&& r)
{
  return RiemannInput<std::decay_t<State>>{l, r};
}

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_RIEMANN_INPUT_HPP