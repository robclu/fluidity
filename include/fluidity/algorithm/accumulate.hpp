//==--- fluidity/algorithm/accumulate.hpp ------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  accumulate.hpp
/// \brief This file implements accumulation by addition.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_ACCUMULATE_HPP
#define FLUIDITY_ALGORITHM_ACCUMULATE_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {

/// Accumulates the \p values, returning their sum.
/// \param[in] v  The current sum of the values.
/// \param[in] vs The rest of the values to accumulate.
/// \tparam    V  The type of the value.
/// \tparam    Vs The types of the rest of the values.
template <typename V, typename... Vs>
fluidity_host_device constexpr auto accumulate(V&& v, Vs&&... vs)
{
  return v + sizeof...(Vs) == 0 ? 0 : accumulate(std::forward<Vs>(vs)...);
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_ACCUMULATE_HPP