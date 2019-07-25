//==--- fluidity/state/state_initialization.hpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_initialization.hpp
/// \brief This file defines the interface for setting state data.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_STATE_STATE_INITIALIZATION_HPP
#define FLUIDITY_STATE_STATE_INITIALIZATION_HPP

#include "cuda/state_initialization.cuh"
#include <fluidity/traits/device_traits.hpp>

namespace fluid {
namespace state {

/// This function sets the state data which is iterated over using the \p states
/// iterator. If an iterated cell is inside the \p levelset then the data for
/// the cell is set using the value of \p state. The \p eos is necessary in the
/// case that the \p states and the \p state type differs and the \p state must
/// then be converted to the type of the state iterated over by \p states. This
/// overload is only enabled if the state iterator is a cpu iterator.
/// \param[in] states     Iterator to the state data.
/// \param[in] levelset   Iterator to the levelset data.
/// \param[in] eos        The equation of state for the material.
/// \tparam    SIT        The type of the state iterator.
/// \tparam    LSIT       The type of the wavespeed iterator.
/// \tparam    State      The type of the state to set with.
/// \tparam    Eos        The type of the equation of state.
template <typename SIT                          ,
          typename LSIT                         ,
          typename State                        ,
          typename Eos                          ,
          ::fluid::traits::cpu_enable_t<SIT> = 0>
void set_states(SIT&& states, LSIT&& levelset, State&& state, Eos&& eos) {
  // Call CPU implementation ...
}

/// This function sets the state data which is iterated over using the \p states
/// iterator. If an iterated cell is inside the \p levelset then the data for
/// the cell is set using the value of \p state. The \p eos is necessary in the
/// case that the \p states and the \p state type differs and the \p state must
/// then be converted to the type of the state iterated over by \p states. This
/// overload is only enabled if the state iterator is a device iterator.
/// \param[in] states     Iterator to the state data.
/// \param[in] levelset   Iterator to the levelset data.
/// \param[in] eos        The equation of state for the material.
/// \tparam    SIT        The type of the state iterator.
/// \tparam    LSIT       The type of the wavespeed iterator.
/// \tparam    State      The type of the state to set with.
/// \tparam    Eos        The type of the equation of state.
template <typename SIT                          ,
          typename LSIT                         ,
          typename State                        ,
          typename Eos                          ,
          ::fluid::traits::gpu_enable_t<SIT> = 0>
void set_states(SIT&& states, LSIT&& levelset, State&& state, Eos&& eos) {
  cuda::set_states(std::forward<SIT>(states)   ,
                   std::forward<LSIT>(levelset),
                   std::forward<State>(state)  ,
                   std::forward<Eos>(eos)      );
}

}} // namespace fluid::state

#endif // FLUIDITY_STATE_STATE_INITIALIZATION_HPP
