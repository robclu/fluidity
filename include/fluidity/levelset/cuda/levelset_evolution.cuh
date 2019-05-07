//==--- fluidity/levelset/cuda/first_order_evolution.cuh --- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  first_order_evolution.cuh
/// \brief This file defines cuda functionality for evolving the levelset.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_CUDA_EVOLUTION_HPP
#define FLUIDITY_LEVELSET_CUDA_EVOLUTION_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace levelset {
namespace cuda     {

template <typename Evolver   ,
          typename InIt      ,
          typename OutIt     ,
          typename Velocities,
          typename T         >
fluidity_global void
evolve_levelset_impl(Evolver    evolver   ,
                     InIt       in        ,
                     OutIt      out       ,
                     Velocities velocities,
                     T          dt        )
{
  Evolver::evolve(in, out, velocities, dt);
}

template <typename Evolver   ,
          typename InData    ,
          typename OutData   ,
          typename Velocities,
          typename T         >
void evolve_levelset(Evolver&&    evolver   ,
                     InData&&     in        ,
                     OutData&&    out       ,
                     Velocities&& v         ,
                     T            dt        )
{
  auto threads = exec::get_thread_sizes(in);
  auto blocks  = exec::get_block_sizes(in, threads);

  evolve_levelset_impl<<<blocks, threads>>>(evolver, in, out, v, dt);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::cuda


#endif // FLUIDITY_LEVELSET_CUDA_EVOLUTION_HPP
