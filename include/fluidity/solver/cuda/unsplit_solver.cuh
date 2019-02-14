//==--- fluidity/solver/unsplit_solver.cuh ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unsplit_solver.cuh
/// \brief This file defines a cuda implementation for a dimensionally-unsplit
///        solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_UNSPLIT_SOLVER_CUH
#define FLUIDITY_SOLVER_UNSPLIT_SOLVER_CUH

#include "../boundary_loader.hpp"
#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace solver {
namespace detail {
namespace cuda   {

/// Wrapper function to invoke a solving kernel for solving in the \p dim
/// dimension.
/// \param[in] in       The input data to use to update.
/// \param[in] out      The output data to write to after updating.
/// \param[in] dtdh     Scaling factor for the update.
/// \tparam    S        The type of the solver.
/// \tparam    IT       The type of the mulri dimensional iterator.
/// \tparam    Mat      The type of the material.
/// \tparam    T        The type of the scaling factor.
template <typename S, typename IT, typename Mat, typename T>
fluidity_global void
solve(IT in, IT out, Mat mat, T dtdh, BoundarySetter setter)
{
  S::invoke(in, out, mat, dtdh, setter);
}

/// Updater function for updating the simulation using the GPU. This invokes 
/// the updating CUDA kernel for the simulation.
/// 
/// \param[in] solver   The split solver implementation.
/// \param[in] in       The input data to use to update.
/// \param[in] out      The output data to write to after updating.
/// \param[in] dtdh     Scaling factor for the update.
/// \tparam    Solver   The type of the solver.
/// \tparam    It       The type of the mulri dimensional iterator.
/// \tparam    Mat      The type of the material.
/// \tparam    T        The type of the scaling factor.
template <typename Solver, typename IT, typename Mat, typename T>
void solve_impl_unsplit(Solver&&               solver,
                        IT&&                   in    ,
                        IT&&                   out   ,
                        Mat&&                  mat   ,
                        T                      dtdh  ,
                        const BoundarySetter&  setter)
{
  using solver_t = std::decay_t<Solver>;
  solve<solver_t><<<solver.block_sizes(), solver.thread_sizes()>>>
  (
    in, out, mat, dtdh, setter
  );
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}}} // namespace fluid::sim::detail::cuda

#endif // FLUIDITY_SOLVER_UNSPLIT_SOLVER_CUH