//==--- fluidity/traits/iterator_traits.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  iterator_traits.hpp
/// \brief This file defines traits for iterators.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_TRAITS_ITERATOR_TRAITS_HPP
#define FLUIDITY_TRAITS_ITERATOR_TRAITS_HPP

#include <fluidity/material/material_iterator.hpp>
#include <fluidity/iterator/multidim_iterator_fwd.hpp>
#include <type_traits>

namespace fluid  {
namespace traits {

/// Defines the underlying value type for the iterator.
/// \tparam Iterator The iterator get the value type for.
template <typename Iterator>
using iter_value_t = typename std::decay_t<Iterator>::value_t;

namespace detail {

/// Struct which is specialized for Multidimensional iterators.
/// \tparam T The type to determine if is a Multidimensional iterator.
template <typename T>
struct IsMultidimensional {
  /// Defines that the type is not multidimensional.
  static constexpr auto value = false;
};

/// Specialization for multidimensional iterators.
/// \tparam T         The type of the data to iterate over.
/// \tparam DimInfo   Information for the dimensions.
/// \tparam Exec      The execution policy for the iterator.
template <typename T, typename DimInfo, typename Exec>
struct IsMultidimensional<MultidimIterator<T, DimInfo, Exec>> {
  /// Defines that the class is multidimensional.
  static constexpr auto value = true;
};

/// Struct which is specialized for material iterators.
/// \tparam T The type to determine if is a material iterator.
template <typename T>
struct IsMaterialIterator {
  /// Defines that the type is not multidimensional.
  static constexpr auto value = false;
};

/// Specialization for material iterators.
/// \tparam EquationOfState   The equation of state for the material.
/// \tparam LevelsetIterator  The type of the iterator over the levelset data.
/// \tparam StateIterator     The type of the iterator over the state data.
template <
  typename EquationOfState ,
  typename LevelsetIterator,
  typename StateIterator
>
struct IsMaterialIterator<
  material::MaterialIterator<EquationOfState, LevelsetIterator, StateIterator>
> {
  /// Defines that the class is a material iterator.
  static constexpr auto value = true;
};

} // namespace detail

/// Returns true if the type T is a multidimensional iterator, otherwise returns
/// false.
/// \tparam T The type to determine if is a multi dimensional iterator.
template <typename T>
static constexpr auto is_multidim_iter_v 
  = detail::IsMultidimensional<std::decay_t<T>>::value;

/// Returns true if the type T is a material iterator, otherwise returns
/// false.
/// \tparam T The type to determine if is a material iterator.
template <typename T>
static constexpr auto is_material_iter_v 
  = detail::IsMaterialIterator<std::decay_t<T>>::value;

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

/// Defines a valid type used for enabling specializations for a 1 dimensional
/// iterator.
/// \tparam T The type to check for 1d enabling.
template <typename T>
using it_1d_enable_t = std::enable_if_t<std::decay_t<T>::dimensions == 1, int>;

/// Defines a valid type used for enabling specializations for a 2 dimensional
/// iterator.
/// \tparam T The type to check for 2d enabling.
template <typename T>
using it_2d_enable_t = std::enable_if_t<std::decay_t<T>::dimensions == 2, int>;

/// Defines a valid type used for enabling specializations for a 3 dimensional
/// iterator.
/// \tparam T The type to check for 3d enabling.
template <typename T>
using it_3d_enable_t = std::enable_if_t<std::decay_t<T>::dimensions == 3, int>;

}} // namespace fluid::traits

#endif // FLUIDITY_TRAITS_ITERATOR_TRAITS_HPP
