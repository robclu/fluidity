//==--- fluidity/math/math.hpp ----------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  math.hpp
/// \brief This file defines math functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATH_MATH_HPP
#define FLUIDITY_MATH_MATH_HPP

#include <fluidity/utility/portability.hpp>
#include <type_traits>

namespace fluid  {
namespace math   {
namespace detail {

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Specialization for an unsigned type.
/// \param[in]  x      The value to get the sign of.
/// \param[in]  select Used to overload the function for non signed type.
/// \tparam     T      The type of the data.
template <typename T>
fluidity_host_device constexpr inline T signum(T x, std::false_type /*select*/)
{
  return T(0) < x;
}

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Specialization for an signed type.
/// \param[in]  x      The value to get the sign of.
/// \param[in]  select Used to overload the function for non signed type.
/// \tparam     T      The type of the data.
template <typename T>
fluidity_host_device constexpr inline T signum(T x, std::true_type /*select*/)
{
  return (T(0) < x) - (x < T(0));
}

} // namespace detail

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Interface function.
/// \param[in]  x  The value to get the sign of.
/// \tparam     T  The type of the data.
template <typename T>
fluidity_host_device constexpr inline T signum(T x)
{
  return detail::signum(x, std::is_signed<T>());
}

}} // namespace fluid::math
 
#endif // FLUIDITY_MATH_MATH_HPP