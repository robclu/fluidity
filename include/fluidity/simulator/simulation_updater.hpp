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
#include <fluidity/execution/execution_policy.hpp>

namespace fluid {
namespace sim   {

/// Updater function for updating the simulation. This overload is only enabled
/// which the input and output iterators are for CPU execution.
/// \param[in] in           The input data to use to update.
/// \param[in] out          The output data to write to after updating.
/// \param[in] solver       The solver which updates the states.
/// \param[in] mat          The material for the system.
/// \param[in] dtdh         Scaling factor for the update.
/// \param[in] thread_sizes The number of threads in each block.
/// \param[in] block_sizes  The number of blocks in the grid.
/// \tparam    Iterator     The type of the mulri dimensional iterator.
/// \tparam    Solver       The type of the solver.
/// \tparam    Material     The type of the material for the system.
/// \tparam    T            The type of the scaling factor.
/// \tparam    SizeInfo     The type of the size information.
template < typename Iterator
         , typename Solver
         , typename Material
         , typename T
         , typename SizeInfo
         , exec::cpu_enable_t<Iterator> = 0>
void update(Iterator in          ,
            Iterator out         ,
            Solver   solver      ,
            Material mat         ,
            T        dtdh        ,
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
/// \param[in] in           The input data to use to update.
/// \param[in] out          The output data to write to after updating.
/// \param[in] solver       The solver which updates the states.
/// \param[in] mat          The material for the system.
/// \param[in] dtdh         Scaling factor for the update.
/// \param[in] thread_sizes The number of threads in each block.
/// \param[in] block_sizes  The number of blocks in the grid.
/// \tparam    Iterator     The type of the mulri dimensional iterator.
/// \tparam    Solver       The type of the solver.
/// \tparam    Material     The type of the material for the system.
/// \tparam    T            The type of the scaling factor.
/// \tparam    SizeInfo     The type of the size information.
template < typename Iterator
         , typename Solver
         , typename Material
         , typename T
         , typename SizeInfo
         , exec::gpu_enable_t<Iterator> = 0>
void update(Iterator in          ,
            Iterator out         ,
            Solver   solver      ,
            Material mat         ,
            T        dtdh        ,
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

/// Sets the wavespeed values pointed to by the \p wavespeed_it using the state
/// data pointed to by the \p state_it iterator.
/// \param[in] state_it     Iterator to the state data.
/// \param[in] wavespeed_it Iterator to the wavespeed data.
/// \param[in] mat          The material for the system.
/// \tparam    StateIt      The type of the state iterator.
/// \tparam    WsIt         The type of the wavespeed iterator.
/// \tparam    Material     The type of the material.
template < typename StateIt
         , typename WsIt
         , typename Material
         , exec::cpu_enable_t<StateIt> = 0>
void set_wavespeeds(StateIt&& state_it, WsIt&& wavespeed_it, Material&& mat)
{
  // Call CPU implementation ...
}

/// Sets the wavespeed values pointed to by the \p wavespeed_it using the state
/// data pointed to by the \p state_it iterator.
/// \param[in] state_it     Iterator to the state data.
/// \param[in] wavespeed_it Iterator to the wavespeed data.
/// \param[in] mat          The material for the system.
/// \tparam    StateIt      The type of the state iterator.
/// \tparam    WsIt         The type of the wavespeed iterator.
/// \tparam    Material     The type of the material.
template < typename StateIt
         , typename WsIt
         , typename Material
         , exec::gpu_enable_t<StateIt> = 0>
void set_wavespeeds(StateIt&& state_it, WsIt&& wavespeed_it, Material&& mat)
{
  detail::cuda::set_wavespeeds(std::forward<StateIt>(state_it) ,
                               std::forward<WsIt>(wavespeed_it),
                               std::forward<Material>(mat)     );
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_SIMULATION_UPDATER_HPP