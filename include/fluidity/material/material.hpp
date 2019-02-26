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

/// The Material class stores the material implementation (i.e essentially the
/// EOS) as well as the level set data which defines where the material is
/// valid (where the levelset is positive).
/// \tparam MatImpl   The implementation which defines the material and the EOS.
/// \tparam Levelset  The type of the levelset which defines the region.  
template <typename MatImpl, typename Levelset>
class Material {
  /// Defines the type of the material implementation.
  using mat_impl_t = std::decay_t<MatImpl>;
  /// Defines the type of the levelset for the material.
  using levelset_t = std::decay_t<Levelset>;

 public:
  /// Constructor which creates a material and the domain for it.
  Material(mat_impl_t&& mat_impl, levelset_t&& levelset)
  : _levelset(std::forward<levelset_t>(levelset)),
    _mat_impl(std::forward<mat_impl_t>(mat_impl))) {}
};

} // namespace material
} // namespace fluid

#endif // FLUIDITY_MATERIAL_IDEAL_GAS_HPP