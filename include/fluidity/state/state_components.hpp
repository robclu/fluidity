//==--- fluidity/state/state_components.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state_components.hpp
/// \brief This file defines aliases for all possible state components, as well
///        as user defined literals to create them.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_STATE_STATE_COMPONENTS_HPP
#define FLUIDITY_STATE_STATE_COMPONENTS_HPP

#include "state_component.hpp"

namespace fluid      {
namespace state      {
namespace components {

/// Defines an element for the density.
using density_t = decltype("density"_component);
/// Defines an element for the pressure.
using pressure_t = decltype("pressure"_component);
/// Defines an element for the x velocity.
using v_x_t      = decltype("v_x"_component);
/// Defines an element for the y velocity.
using v_y_t      = decltype("v_y"_component);
/// Defines an element for the z velocity.
using v_z_t      = decltype("v_z"_component);

/// User defined literal to create a density state element with value \p v.
/// \param[in] v The value of the density.
auto operator "" _rho(long double v)
{
    return density_t(v);
}

/// User defined literal to create a pressure state element with value \p v.
/// \param[in] v The value of the pressure.
auto operator "" _p(long double v)
{
    return pressure_t(v);
}

/// User defined literal to create an x velocity state element with value \p v.
/// \param[in] v The value of the velocity.
auto operator "" _v_x(long double v)
{
    return v_x_t(v);
}

/// User defined literal to create an y velocity state element with value \p v.
/// \param[in] v The value of the velocity.
auto operator "" _v_y(long double v)
{
    return v_y_t(v);
}

/// User defined literal to create an z velocity state element with value \p v.
/// \param[in] v The value of the velocity.
auto operator "" _v_z(long double v)
{
    return v_z_t(v);
}

} // namespace components

}} // namespace fluid::state 

#endif // FLUIDITY_STATE_STATE_COMPONENTS_HPP