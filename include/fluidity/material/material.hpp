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

#include "material_traits.hpp"
#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace material {

/// The Material class defines functionality for a material, which is
/// essentially the equation of state which defines the material, and a levelset
/// which defines where the material is and is not valid.
///
/// \tparam Eos       The equation of state for the material implementation.
/// \tparam Levelset  The type of the levelset which defines the region.  
template <typename Eos, typename Levelset>
class Material : public Eos {
 private:
  /// Defines the traits for the material.
  using traits_t   = MaterialTraits<Material<Eos, Levelset>>;
  /// Defines the type of the equation of state.
  using eos_t      = typename traits_t::eos_t;
  /// Defines the type of the levelset.
  using levelset_t = typename traits_t::levelset_t;

 public:
  /// Inherit constructors from the equation of state so that the material can
  /// be created in the same way as the equation of state.
  using eos_t::eos_t;

  /// Default constructor for the material.
  Material() = default;

  /// Constructor which creates a material and the domain for it.
  Material(levelset_t&& levelset)
  : _levelset(std::move(levelset)), _is_set{true} {
    traits_t::assert_eos();
  }

  /// Returns a const reference to the levelset for the material.
  auto levelset() const -> const levelset_t& {
    return _levelset;
  }

  /// Returns a reference to the levelset for the material.
  auto levelset() -> levelset_t& {
    return _levelset;
  }

  /// Returns a const reference to the equation of state for the material.
  auto eos() const -> const eos_t& {
    return _eos;
  }

  /// Returns a reference to the of equation of state for the material.
  auto eos() -> eos_t& {
    return _eos;
  }

  /// Returns true if the material levelset data has been initialized.
  auto is_initialized() const -> bool {
    return _is_set;
  }

  /// Initializes the levelset using the predicate \p predicate.
  /// \param[in] pred      The predicate to use to initialize the levelset.
  /// \tparam    Predicate The type of predicate.
  template <typename Predicate>
  auto init_levelset(Predicate&& pred) -> void {
    _levelset.initialize(std::forward<Predicate>(pred));
    _is_set = true;
  }

 private:
  levelset_t _levelset;       //!< The levelset for the material.
  eos_t      _eos;            //!< The equation of state for the material.
  bool       _is_set = false; //!< If the material has be set.
};

} // namespace material
} // namespace fluid

#endif // FLUIDITY_MATERIAL_MATERIAL_HPP
