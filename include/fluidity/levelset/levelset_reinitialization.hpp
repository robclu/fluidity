//==--- fluidity/levelset/levelset_reinitialization.hpp ---- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_initialization.hpp
/// \brief This file defines the interface for invoking reinitialization of a
///        levelset. 
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_LEVELSET_REINITIALIZATION_HPP
#define FLUIDITY_LEVELSET_LEVELSET_REINITIALIZATION_HPP

#include "cuda/levelset_reinitialization.cuh"
#include <fluidity/execution/execution_policy.hpp>

namespace fluid    {
namespace levelset {

/// This re-initializes the \p levelset data to ensure that it is a signed
/// distance function. The specific method which is used for the initialization
/// is an implementation detail of the \p initializer. This overload is only
/// eneabled if the iterator for the levelset data has a CPU execution policy.
/// \param[in] initializer The implementation of the initializer.
/// \param[in] levelset    The levelset to reinitialize.
/// \tparam    Init        The type of the initializer.
/// \tparam    Levelset    The type of the levelset.
template <typename Init, typename Levelset, exec::cpu_enable_t<Levelset> = 0>
void reinitialize(Init&& initializer, Levelset&& levelset)
{
  // Call CPU implementation ...
}

/// This re-initializes the \p levelset data to ensure that it is a signed
/// distance function. The specific method which is used for the initialization
/// is an implementation detail of the \p initializer. This overload is only
/// eneabled if the iterator for the levelset data has a CPU execution policy.
/// \param[in] initializer The implementation of the initializer.
/// \param[in] levelset    The levelset to reinitialize.
/// \tparam    Init        The type of the initializer.
/// \tparam    Levelset    The type of the levelset.
template <typename Init, typename Levelset, exec::cpu_enable_t<Levelset> = 0>
void reinitialize(Init&& initializer, Levelset&& levelset)
{
  cuda::reinit_levelset(std::forward<Init>(initializer),
                        std::forward<Levelset>(levelset));
}

}} // namespace fluid::levelset

#endif // FLUIDITY_LEVELSET_LEVELSET_REINITIALIZATION_HPP