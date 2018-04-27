//==--- fluidity/simulator/simulation_updater.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_updater.hpp
/// \brief This file defines a class which updates a simulation, and the
///        implementation is specialized for CPU and GPU execution policies.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP
#define FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP

#include "simulation_updater.cuh"

namespace fluid {
namespace sim   {

/// Updater function for updating the simulation. This overload is only enabled
/// which the input and output iterators are for CPU execution.
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
         , std::enable_if_t<
             exec::is_cpu_policy_v<
               typename std::decay_t<Iterator>::exec_policy_t
             >, int> = 0
         >
void update(Iterator in          ,
            Iterator out         ,
            Solver   solver      ,
            Material mat         ,
            T          dtdh        ,
            SizeInfo thread_sizes,
            SizeInfo block_sizes )
{
  // Call CPU implementation ...
}

/// Updater function for updating the simulation. This overload is only enabled
/// which the input and output iterators are for GPU execution.
/// 
/// This simply forwards all the arguments onto the cuda implementation of the
/// updating function.
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
         , std::enable_if_t<
             exec::is_gpu_policy_v<
               typename std::decay_t<Iterator>::exec_policy_t
             >, int> = 0
         >
void update(Iterator in          ,
            Iterator out         ,
            Solver   solver      ,
            Material mat         ,
            T          dtdh        ,
            SizeInfo thread_sizes,
            SizeInfo block_sizes )
{
  detail::cuda::update(
    std::forward<Iterator>(in)          ,
    std::forward<Iterator>(out)         ,
    std::forward<Solver>(solver)        ,
    std::forward<Material>(mat)         ,
    dtdh                                ,
    std::forward<SizeInfo>(thread_sizes),
    std::forward<SizeInfo>(block_sizes) );
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP