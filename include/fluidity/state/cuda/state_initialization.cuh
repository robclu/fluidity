//==--- fluidity/state/cuda/state_initialization.cuh ------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_initialization.cuh
/// \brief This file defines cuda functionality which allows state data to be
///        initialized.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_STATE_CUDA_STATE_INITIALIZATION_HPP
#define FLUIDITY_STATE_CUDA_STATE_INITIALIZATION_HPP

#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/levelset/levelset.hpp>
#include <fluidity/state/state_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace state  {
namespace cuda   {
namespace detail {

using namespace fluid::state::traits;

/// Sets the \p out state using the \p in state, using the \p eos to convert if
/// the types differ. This overload is enabled when the output state is of
/// primitive form.
/// \param[in] out  The state to set.
/// \param[in] in   The state to use to set.
/// \param[in] eos  The equation of state for the material.
/// \tparam    S1   The type of the output state.
/// \tparam    S2   The type of the setter state.
/// \tparam    Eos  The type of the equation of state.
template <typename S1, typename S2, typename Eos, prim_enable_t<S1> = 0>
fluidity_host_device void set_state(S1& out, S2& in, Eos& eos)
{
  out = in.primitive(eos);
}

/// Sets the \p out state using the \p in state, using the \p eos to convert if
/// the types differ. This overload is enabled when the output state is of
/// conservative form.
/// \param[in] out  The state to set.
/// \param[in] in   The state to use to set.
/// \param[in] eos  The equation of state for the material.
/// \tparam    S1   The type of the output state.
/// \tparam    S2   The type of the setter state.
/// \tparam    Eos  The type of the equation of state.
template <typename S1, typename S2, typename Eos, cons_enable_t<S1> = 0>
fluidity_host_device void set_state(S1& out, S2& in, Eos& eos)
{
  out = in.conservative(eos);
}

/// Implementation kernel to set the \p state_it state data using the \p
/// levelset_it for cells which are inside the \p levelset_t.
/// \param[in] state_it     An iterator over the material state data.
/// \param[in] levelset_it  An iterator over the material levelset data.
/// \param[in] state        The state to use to set the data.        
/// \param[in] eos          The equation of state for the material.
/// \tparam    SIT          The type of the state iterator.
/// \tparam    LSIT         The type of the levelset iterator.
/// \tparam    State        The type of the setter state.
/// \tparam    Eos          The type of the equation of state.
template <typename SIT, typename LSIT, typename State, typename Eos>
fluidity_global void
set_states_impl(SIT state_it, LSIT levelset_it, State state, Eos eos)
{
  // TODO: Check if this will be faster in shared memory ...
  unrolled_for<std::decay_t<SIT>::dimensions>([&] (auto i)
  {
    state_it.shift(flattened_id(i), i);
    levelset_it.shift(flattened_id(i), i);
  });

  if (levelset::inside(levelset_it))
  {
    detail::set_state(*state_it, state, eos);
  }
}

} // namespace detail

/// Sets the \p state_it state data for a material using the \p levelset_it
/// levelset iterator for the material, to set cells which are inside the
/// material using the value defined by the \p state.
/// \param[in] state_it     An iterator over the material state data.
/// \param[in] levelset_it  An iterator over the material levelset data.
/// \param[in] state        The state to use to set the data.        
/// \param[in] eos          The equation of state for the material.
/// \tparam    SIT          The type of the state iterator.
/// \tparam    LSIT         The type of the levelset iterator.
/// \tparam    State        The type of the setter state.
/// \tparam    Eos          The type of the equation of state.
template <typename SIT, typename LSIT, typename State, typename Eos>
void set_states(SIT&& state_it, LSIT&& levelset_it, State&& state, Eos&& eos)
{
  using state_it_t    = std::decay_t<SIT>;
  using levelset_it_t = std::decay_t<LSIT>;

  static_assert(
    state_it_t::dimensions == levelset_it_t::dimensions,
    "Number of dimensions for state and levelset iterators do not match!");

  auto threads = exec::get_thread_sizes(state_it);
  auto blocks  = exec::get_block_sizes(state_it, threads);
  detail::set_states_impl<<<blocks, threads>>>(state_it   ,
                                               levelset_it,
                                               state      ,
                                               eos        );
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} /// namespace fluid::state::cuda

#endif // FLUIDITY_STATE_CUDA_STATE_INITIALIZATION_HPP