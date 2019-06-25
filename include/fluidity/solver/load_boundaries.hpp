//==--- fluidity/solver/load_boundaries.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  wavespeed_initialization.hpp
/// \brief This file defines the interface for wavespeed initialization.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_LOAD_BOUNDARIES_HPP
#define FLUIDITY_SOLVER_LOAD_BOUNDARIES_HPP

#include "cuda/load_boudaries.hpp"

namespace fluid {
namespace solver {

/// Loads the bounday data for the global iterator \p states, using the boundary
/// loader defined for the \p solver. This overload is only enabled if the \p
/// states iterator is a CPU iterator.
/// \param[in] solver The solver for the system.
/// \param[in] states Iterator to the state data.
/// \param[in] setter The setter for the boundary data. 
/// \tparam    Solver The type of the solver.
/// \tparam    IT     The type of the state iterator.
template <typename Solver, typename IT, exec::cpu_enable_t<IT> = 0>
void load_boundaries(Solver&& sover, IT&& states, const BoundarySetter& setter)
{
  // Call CPU implementation ...
}

/// Loads the bounday data for the global iterator \p states, using the boundary
/// loader defined for the \p solver. This overload is only enabled if the \p
/// states iterator is a GPU iterator.
/// \param[in] solver The solver for the system.
/// \param[in] states Iterator to the state data.
/// \param[in] setter The setter for the boundary data. 
/// \tparam    Solver The type of the solver.
/// \tparam    IT     The type of the state iterator.
template <typename Solver, typename IT, exec::gpu_enable_t<IT> = 0>
void load_boundaries(Solver&& sover, IT&& states, const BoundarySetter& setter)
{
  detail::cuda::load_boundaries(std::forward<Solver>(solver),
                                std::forward<IT>(states)    ,
                                setter                      );
}

}} // namespace fluid::solver

#define FLUIDITY_SOLVER_LOAD_BOUNDARIES_HPP
