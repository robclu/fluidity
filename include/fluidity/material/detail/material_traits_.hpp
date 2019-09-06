//==--- fluidity/material/detail/material_traits_.hpp- --------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_traits_.hpp
/// \brief This file defines utilities for type traits for material.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_DETAIL_MATERIAL_TRAITS__HPP 
#define FLUIDITY_MATERIAL_DETAIL_MATERIAL_TRAITS__HPP

#include <fluidity/container/tuple.hpp>

namespace fluid    {
namespace material {
namespace detail   {

/// Defines a struct to determine if a container is multi material.
/// \tparam Materials The type(s) of the material(s).
template <typename Materials>
struct IsMultiMaterial {
  /// Defines that this is not multi-material.
  static constexpr auto value = false;
};

/// Specialization for tuples, which checks if the inner classes are materials.
/// \tparam Materials The type(s) of the material(s).
template <typename... Materials>
struct IsMultiMaterial<Tuple<Materials...>> {
  /// Defines that this is multi-material.
  /// TODO: Change this to check the variadic pack for material conformatlity.
  static constexpr auto value = true;
}; 

}}} // namespace fluid::material

#endif // FLUIDITY_MATERIAL_DETAIL_MATERIAL_TRAITS__HPP

