//==--- fluidity/material/material_traits.hpp- ---------------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_traits.hpp
/// \brief This file defines type traits for material.s
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_MATERIAL_TRAITS_HPP 
#define FLUIDITY_MATERIAL_MATERIAL_TRAITS_HPP

#include "detail/material_traits_.hpp"
#include "eos_traits.hpp"

namespace fluid    {
namespace material {

/// Trait which returns true if T is a container of multiple materials.
/// \tparam T The type to check if multi material.
template <typename T>
static constexpr auto is_mmaterial_v = detail::IsMultiMaterial<T>::value;

/// Defines a valid type if T is a multimaterial container.
/// \tparam T The type to base the enabling on.
template <typename T>
using mmaterial_enable_t = std::enable_if_t<is_mmaterial_v<T>, int>;

/// Defines a valid type if T is not a multimaterial container.
/// \tparam T The type to base the enabling on.
template <typename T>
using non_mmaterial_enable_t = std::enable_if_t<!is_mmaterial_v<T>, int>;

//==--- [Forward declarations] ---------------------------------------------==//

/// The Material class defines functionality for a material, which is
/// essentially the equation of state which defines the material, and a levelset
/// which defines where the material is and is not valid.
///
/// \tparam Eos       The equation of state for the material implementation.
/// \tparam Levelset  The type of the levelset which defines the region.  
template <typename Eos, typename Levelset> class Material;

//==--- [Default material traits] ------------------------------------------==//

/// The Material traits class defines traits for the material, which can be
/// specialized for different material implementations.
/// \tparam Material The material to specialize the traits for.
template <typename Material>
struct MaterialTraits {};

//==--- [Specialziation] ---------------------------------------------------==//

/// Specialization of material traits for the Material class.
/// \tparam Eos      The equation of state for the material.
/// \tparam Levelset The levelset for the material.
template <typename Eos, typename Levelset>
struct MaterialTraits<Material<Eos, Levelset>> {
  /// Defines the type of the equation of state.
  using eos_t            = std::decay_t<Eos>;
  /// Defines the type of a const equation of state.
  using const_eos_t      = const std::decay_t<Eos>;
  /// Defines the type of the levelset for the material.
  using levelset_t       = std::decay_t<Levelset>;
  /// Defines the type of a const levelset.
  using const_levelset_t = const std::decay_t<Levelset>;

  /// Asserts if the equation of state is not an equation of state.
  static constexpr auto assert_eos() -> void {
    static_assert(
      is_eos_v<Eos>,
      "Equation of state for material must conform to EquationOfState interface"
    );
  }
};

}} // namespace fluid::material

#endif // FLUIDITY_MATERIAL_MATERIAL_TRAITS_HPP

