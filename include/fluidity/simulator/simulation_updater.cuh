//==--- fluidity/simulator/simulation_updater.cuh ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_updater.cuh
/// \brief This file defines a class which updates a simulation using the GPU.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH
#define FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH

#include <fluidity/utility/debug.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace sim    {
namespace detail {
namespace cuda   {

/// Implementation of the updating function. This simply invokes the solving
/// function of the solver with the input and output data and the simulation
/// paramters.
/// \param[in] in     The iterator over the input data for the simulation.
/// \param[in] out    The iterator over the output data for the simulation.
/// \param[in] dtdh   The scaling factor for state update.
/// \param[in] solver The solver which solves and updates the simulation.
/// \tparam    It     The type of the iterators.
/// \tparam    T      The type of the scaling factor.
/// \tparam    Solver The type of the solver.
template <typename It, typename M, typename T, typename Solver>
fluidity_global void update_impl(It in, It out, M mat, T dtdh, Solver solver)
{
  solver.solve(in, out, mat, dtdh);
}

/// Updater function for updating the simulation using the GPU. This invokes 
/// the updating CUDA kernel for the simulation.
/// 
/// \param[in] multi_iterator_in The input data to use to update.
/// \param[in] multi_itertor_out The output data to write to after updating.
/// \param[in] dtdh              Scaling factor for the update.
/// \param[in] solver            The solver which updates the states.
/// \param[in] thread_sizes      The number of threads in each block.
/// \param[in] block_sizes       The number of blocks in the grid.
/// \tparam    Iterator          The type of the mulri dimensional iterator.
/// \tparam    T                 The type of the scaling factor.
/// \tparam    Solver            The type of the solver.
/// \tparam    SizeInfo          The type of the size information.
template < typename Iterator
         , typename Solver
         , typename Material
         , typename T
         , typename SizeInfo
         >
void update(Iterator&& in          ,
            Iterator&& out         ,
            Solver&&   solver      ,
            Material&& mat         ,
            T          dtdh        ,
            SizeInfo&& thread_sizes,
            SizeInfo&& block_sizes )
{
#if defined(__CUDACC__)
  update_impl<<<block_sizes, thread_sizes>>>(in, out, mat, dtdh, solver);
  fluidity_check_cuda_result(cudaDeviceSynchronize()); 
#endif // __CUDACC__
}

}}}} // namespace fluid::sim::detail::cuda

#endif // FLUIDITY_SIMULATOR_SIMULATION_UPDATER_CUH