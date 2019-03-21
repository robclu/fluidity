//==--- fluidity/material/material.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material.hpp
/// \brief This file defines a material class which stores the material
///        definition (essentially the equation of state) as well as the
///        levelset data which defines the domain for which the material applies
///        (i.e where the material is valid).
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_MATERIAL_HPP
#define FLUIDITY_MATERIAL_MATERIAL_HPP

//#include "material.hpp"
#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace material {

namespace detail {

/// Defines a struct to determine if a container is multi material.
/// \tparam Materials The type(s) of the material(s).
template <typename Materials>
struct IsMultiMaterial {
  /// Defines that this is not multi-material.
  static constexpr auto value = false;
};


template <typename... Materials>
struct IsMultiMaterial<Tuple<Materials...>> {
  /// Defines that this is multi-material.
  static constexpr auto value = true;
}; 

} // namespace detail

template <typename T>
static constexpr auto is_mmaterial_v = detail::IsMultiMaterial<T>::value;

/// Defines a valid type if T is a multimaterial container.
template <typename T>
using mmaterial_enable_t     = std::enable_if_t<is_mmaterial_v<T>, int>;

/// Defines a valid type if T is not a multimaterial container.
template <typename T>
using non_mmaterial_enable_t = std::enable_if_t<!is_mmaterial_v<T>, int>;

/// The Material class stores the material implementation (i.e essentially the
/// EOS) as well as the level set data which defines where the material is
/// valid (where the levelset is positive).
/// \tparam Eos       The equation of state for the material implementation.
/// \tparam Levelset  The type of the levelset which defines the region.  
template <typename Eos, typename Levelset>
class Material : public Eos {
  /// Defines the type of the material implementation.
  using eos_t      = std::decay_t<Eos>;
  /// Defines the type of the levelset for the material.
  using levelset_t = std::decay_t<Levelset>;

 public:
  /// Inherit constructors from the equation of state so that the material can
  /// be created in the same way as the equation of state.
  using eos_t::eos_t;

  /// Default constructor for the material.
  Material() = default;

  /// Constructor which creates a material and the domain for it.
  Material(levelset_t&& levelset)
  : _levelset(std::forward<levelset_t>(levelset)), _is_set{true} {}

  /// Returns a const reference to the levelset for the material.
  const auto& levelset() const { return _levelset; }

  /// Returns a reference to the levelset for the material.
  auto& levelset() { return _levelset; }

  /// Returns a const reference to the equation of state for the material.
  const auto eos() const { return _eos; }

  /// Returns a reference to the of equation of state for the material.
  auto& eos() { return _eos; }

  /// Returns true if the material levelset data has been initialized.
  bool is_initialized() const { return _is_set; }

  /// Initializes the levelset using the predicate \p pred.
  template <typename Pred>
  void init_levelset(Pred&& pred)
  {
    _levelset.initialize(std::forward<Pred>(pred));
    _is_set = true;
  }

 private:
  levelset_t _levelset;       //!< The levelset for the material.
  eos_t      _eos;            //!< The equation of state for the material.
  bool       _is_set = false; //!< If the material has be set.
};

} // namespace material
} // namespace fluid

#endif // FLUIDITY_MATERIAL_IDEAL_GAS_HPP