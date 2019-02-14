//==--- fluidity/iterator/iterator_traits.hpp -------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  iterator_traits.hpp
/// \brief This file defines traits for iterators.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ITERATOR_ITERATOR_TRAITS_HPP
#define FLUIDITY_ITERATOR_ITERATOR_TRAITS_HPP

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/execution/execution_policy.hpp>
#include <type_traits>


namespace fluid {

/// Forward declaration of a class for multi-dimensional iteration, which allows
/// traits for the iterator to be defined.
/// \tparam T         The type of the data to iterate over.
/// \tparam DimInfo   Information for the dimensions.
/// \tparam Exec      The execution policy for the iterator.
template <typename T,
          typename DimensionInfo = DimInfo<2>,
          typename Exec          = exec::default_type>
struct MultidimIterator;

/// Defines the underlying value type for the iterator.
/// \tparam Iterator The iterator get the value type for.
template <typename Iterator>
using iter_value_t = typename std::decay_t<Iterator>::value_t;

namespace detail {

/// Struct which is specialized for Multidimensional iterators.
/// \tparam T The type to determine if is a Multidimensional iterator.
template <typename T>
struct IsMultidimensional
{
  /// Defines that the type is not multidimensional.
  static constexpr auto value = false;
};

/// Specialization for multidimensional iterators.
/// \tparam T         The type of the data to iterate over.
/// \tparam DimInfo   Information for the dimensions.
/// \tparam Exec      The execution policy for the iterator.
template <typename T, typename DimInfo, typename Exec>
struct IsMultidimensional<MultidimIterator<T, DimInfo, Exec>>
{
  /// Defines that the class is multidimensional.
  static constexpr auto value = true;
};

} // namespace detail

/// Returns true if the type T is a multidimensional iterator, otherwise returns
/// false.
/// \tparam T The type to determine if is a multi dimensional iterator.
template <typename T>
static constexpr auto is_multidim_iter_v = detail::IsMultidimensional<T>::value;

/// Defines a valid type used for enabling of multidimensional specializations.
/// \tparam T The type to check multidimensional functionality for to base the
///           enabling on.
template <typename T>
using multiit_enable_t = std::enable_if_t<is_multidim_iter_v<T>, int>;

/// Defines a valid type used for enabling of non multidimensional  
/// specializations.
/// \tparam T The type to check multidimensional functionality for to base the
///           enabling on.
template <typename T>
using nonmultiit_enable_t = std::enable_if_t<!is_multidim_iter_v<T>, int>;

} // namespace fluid

#endif // FLUIDITY_ITERATOR_ITERATOR_TRAITS_HPP