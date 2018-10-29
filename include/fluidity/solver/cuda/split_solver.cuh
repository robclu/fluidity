//==--- fluidity/solver/split_solver.cuh ------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  split_solver.cuh
/// \brief This file defines the cuda implementation for the split solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_SPLIT_SOLVER_CUH
#define FLUIDITY_SOLVER_SPLIT_SOLVER_CUH

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
/// \tparam    Dim     s The type of the dimension.
template <typename S, typename IT, typename Mat, typename T, typename Dim>
fluidity_global void
solve_dim(IT in, IT out, Mat mat, T dtdh, BoundarySetter setter, Dim dim)
{
  S::invoke(in, out, mat, dtdh, setter, dim);
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
void solve_impl(Solver&&               solver,
                IT&&                   in    ,
                IT&&                   out   ,
                Mat&&                  mat   ,
                T                      dtdh  ,
                const BoundarySetter&  setter)
{
  using iter_t   = std::decay_t<IT>;
  using solver_t = std::decay_t<Solver>;
  unrolled_for<iter_t::dimensions>([&] (auto i)
  {
    constexpr auto dim = Dimension<i>();
    solve_dim<solver_t><<<solver.block_sizes(), solver.thread_sizes()>>>
    (
      in, out, mat, dtdh, setter, dim
    );
    fluidity_check_cuda_result(cudaDeviceSynchronize());
    if (iter_t::dimensions != 1 && i < iter_t::dimensions - 1)
    {
      std::swap(in, out);
    }
  });
}

}}}} // namespace fluid::sim::detail::cuda

#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_CUH