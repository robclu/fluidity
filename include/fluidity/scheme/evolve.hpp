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
#include <fluidity/boundary/boundary_loading.hpp>
#include <fluidity/traits/device_traits.hpp>

namespace fluid  {
namespace scheme {

/// Interface for evolving the \p in_it data using the \p evolver to set the
/// \p out_it data. After calling this method, the \p out_it data will
/// be evolved by \p dt from the \p in_it data, using the method implemented by
/// the \p evolver. 
///
/// This overload is only enabled when the \p in_it iterator is a CPU iterator.
///
/// \param[in] evolver        The type of the evolver.    
/// \param[in] in_it          The input iterator.
/// \param[in] out_it         The output iterator.          
/// \param[in] dt             The time delta for the evolution.
/// \param[in] dh             The size of the spacial resolution.
/// \param[in] boundaries     The boundaries for the evolution.
/// \param[in] func_or_it     An functor/iterator to use for the evaluation.
/// \param[in] args           Additional argumens for the evolution.
/// \tparam    Evolver        The type of the evolver.
/// \tparam    InIterator     The type of the input iterator.
/// \tparam    OutIterator    The type of the output iterator.
/// \tparam    T              The data type for the deltas.
/// \tparam    BoundContainer The type of the boundary container.
/// \tparam    FuncOrIt       The type of the functor/additional iterator.
/// \tparam    Args           Additional argument types.
template <
  typename    Evolver       ,
  typename    InIterator    ,
  typename    OutIterator   ,
  typename    T             , 
  typename    BoundContainer,
  typename    FuncOrIt      ,
  typename... Args          ,
  traits::cpu_enable_t<InIterator> = 0
>
auto evolve(
  Evolver&&        evolver   ,
  InIterator&&     in_it     , 
  OutIterator&&    out_it    ,
  T                dt        ,
  T                dh        ,
  BoundContainer&& boundaries,
  FuncOrIt&&       func_or_it,
  Args&&...        args
) -> void {
  static_assert(
    is_evolver_v<Evolver>,
    "Provided evolver does not conform to the Evolver interface!"
  );
  // TODO: Add implemenation ... 
}

/// Interface for evolving the \p in_it data using the \p evolver to set the
/// \p out_it data. After calling this method, the \p out_it data will
/// be evolved by \p dt from the \p in_it data, using the method implemented by
/// the \p evolver.  
///
/// This overload is only enabled when the \p in_it iterator is a GPU iterator.
///
/// \param[in] evolver        The type of the evolver.    
/// \param[in] in_it          The input iterator.
/// \param[in] out_it         The output iterator.          
/// \param[in] dt             The time delta for the evolution.
/// \param[in] dh             The size of the spacial resolution.
/// \param[in] boundaries     The boundaries for the evolution.
/// \param[in] func_or_it     An functor/iterator to use for the evaluation.
/// \param[in] args           Additional argumens for the evolution.
/// \tparam    Evolver        The type of the evolver.
/// \tparam    InIterator     The type of the input iterator.
/// \tparam    OutIterator    The type of the output iterator.
/// \tparam    T              The data type for the deltas.
/// \tparam    BoundContainer The type of the boundary container.
/// \tparam    FuncOrIt       The type of the functor/additional iterator.
/// \tparam    Args           Additional argument types.
template <
  typename    Evolver       ,
  typename    InIterator    ,
  typename    OutIterator   ,
  typename    T             , 
  typename    BoundContainer,
  typename    FuncOrIt      ,
  typename... Args          ,
  traits::gpu_enable_t<InIterator> = 0
>
auto evolve(
  Evolver&&        evolver   ,
  InIterator&&     in_it     , 
  OutIterator&&    out_it    ,
  T                dt        ,
  T                dh        ,
  BoundContainer&& boundaries,
  FuncOrIt&&       func_or_it,
  Args&&...        args
) -> void {
  static_assert(
    is_evolver_v<Evolver>,
    "Provided evolver does not conform to the Evolver interface!"
  );
  cuda::evolve(
    std::forward<Evolver>(evolver)          ,
    std::forward<InIterator>(in_it)         ,
    std::forward<OutIterator>(out_it)       ,
    dt                                      ,
    dh                                      ,
    std::forward<BoundContainer>(boundaries),
    std::forward<FuncOrIt>(func_or_it)      ,
    std::forward<Args>(args)...
  );
}

}} // namespace fluid::scheme


#endif // FLUIDITY_SCHEME_EVOLVE_HPP
