//==--- fluidity/solver/eikonal_solver.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  eikonal_solver.hpp
/// \brief This file implements the interface for invoking an Eikonal solver
///        on some input and output data.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_EIKONAL_SOLVER_HPP
#define FLUIDITY_SOLVER_EIKONAL_SOLVER_HPP

#include <fluidity/traits/tensor_traits.hpp>

namespace fluid  {
namespace solver {

/// Solves the Eikonal equation with a constant speed function of $f = 1$,
/// using the \p input data as the input data, and writing the results to the
/// \p output data. The \p solver is the solver implementation which is used.
///
/// This overload is enabled when the \p input and \p output data are device
/// tensors.
///
/// \param[in] input  The input data to use to initialize the solve state.
/// \param[in] output The output data to write the results to.
/// \param[in] solver The solver which computes the Eikonal solution.
/// \tparam    D      The type of the input and output data.
/// \tparam    S      The type of the solver.
template <typename D, typename T, typename S, traits::dtensor_enable_t<D> = 0>
fluidity_host_device void eikonal(D&& input, D&& output, T dh, S&& solver) {
  solver.solve(input.multi_iterator(), output.multi_iterator(), dh);
}

/// Solves the Eikonal equation with a constant speed function of $f = 1$,
/// using the \p input data as the input data, and writing the results to the
/// \p output data. The \p solver is the solver implementation which is used.
///
/// This overload is enabled when the \p input and \p output data are device
/// tensors.
///
/// \param[in] input  The input data to use to initialize the solve state.
/// \param[in] output The output data to write the results to.
/// \param[in] solver The solver which computes the Eikonal solution.
/// \tparam    D      The type of the input and output data.
/// \tparam    S      The type of the solver.
template <
  typename Iterator,
  typename T       ,
  typename Solver  ,
  traits::gpu_enable_t<Iterator> = 0
>
fluidity_host_device void eikonal(
  Iterator&& in_it ,
  Iterator&& out_it,
  T          dh    ,
  Solver&&   solver
) {
  solver.solve(
    std::forward<Iterator>(in_it) ,
    std::forward<Iterator>(out_it),
    dh
  );
}


}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_EIKONAL_SOLVER_HPP
