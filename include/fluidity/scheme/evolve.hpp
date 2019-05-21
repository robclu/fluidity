//==--- fluidity/scheme/evolve.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  evolve.hpp
/// \brief This file defines an interface for evolving input data to output
///        data using a specific evolver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_EVOLVER_HPP
#define FLUIDITY_SCHEME_EVOLVER_HPP

#include "cuda/evolve.cuh"
#include "interfaces/evolver.hpp"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid  {
namespace scheme {


/// Interface for evolving the \p in data using the \p evl to set the \p out
/// data. The \p evl evolver must conform to the Evolver interface. This
/// overload is only enabled when the input iterator has a CPU execution policy.
/// \param[in] evl
/// \param[in] in
/// \param[in] out
/// \param[in] dt
/// \param[in] dh
/// \param[in] args
/// \tparam    Evolver
/// \tparam    ItIn
/// \tparam    ItOut
/// \tparam    T
/// \tparam    Args
template <typename    Evl  ,
          typename    ItIn ,
          typename    ItOut,
          typename    T    , 
          typename... Args , exec::cpu_enable_t<ItIn> = 0>
void evolve(Evl&& evl, ItIn&& in, ItOut&& out, T dt, T dh, Args&&... args)
{
  static_assert(is_evolver_v<Evl>,
                "Provided evolver does not conform to the Evolver interface!");

}

/// Interface for evolving the \p it_in data using the \p evolver to set the
/// \p out_it data.
/// \param[in] evl
/// \param[in] in
/// \param[in] out
/// \param[in] dt
/// \param[in] dh
/// \param[in] args
/// \tparam    Evolver
/// \tparam    ItIn
/// \tparam    ItOut
/// \tparam    T
/// \tparam    Args
template <typename    Evl  ,
          typename    ItIn ,
          typename    ItOut,
          typename    T    , 
          typename... Args , exec::gpu_enable_t<ItIn> = 0>
void evolve(Evl&& evl, ItIn&& in, ItOut&& out, T dt, T dh, Args&&... args)
{
  static_assert(is_evolver_v<Evl>,
                "Provided evolver does not conform to the Evolver interface!");
  cuda::evolve(std::forward<Evl>(evl)     ,
               std::forward<ItIn>(in)     ,
               std::forward<ItOut>(out)   ,
               dt                         ,
               dh                         ,
               std::forward<Args>(args)...);
}


}} // namespace fluid::scheme


#endif // FLUIDITY_SCHEME_EVOLVE_HPP
