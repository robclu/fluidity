//==--- fluidity/levelset/levelset_evolution.hpp ----------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_evolution.cuh
/// \brief This file defines cuda functionality for evolving the levelset.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_EVOLUTION_HPP
#define FLUIDITY_LEVELSET_EVOLUTION_HPP

#include "cuda/levelset_evolution.cuh"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid    {
namespace levelset {

template <typename Evolver   ,
          typename InData    ,
          typename OutData   ,
          typename Velocities,
          typename T, exec::cpu_enable_t<InData> = 0>
void evolve_levelset(Evolver&&    evolver   ,
                     InData&&     in        ,
                     OutData&&    out       ,
                     Velocities&& velocities,
                     T            dt        )
{
}

template <typename Evolver   ,
          typename InData    ,
          typename OutData   ,
          typename Velocities,
          typename T, exec::gpu_enable_t<InData> = 0>
void evolve_levelset(Evolver&&    evolver   ,
                     InData&&     in        ,
                     OutData&&    out       ,
                     Velocities&& velocities,
                     T            dt        )
{
  cuda::evolve_levelset(std::forward<Evolver>(evolver)      ,
                        std::forward<InData>(in)            ,
                        std::forward<OutData>(out)          ,
                        std::forward<Velocities>(velocities),
                        dt                                  );
}

}} // namespace fluid::cuda


#endif // FLUIDITY_LEVELSET_EVOLUTION_HPP
