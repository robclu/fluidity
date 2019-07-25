//==--- fluidity/ghost_fluid/riemann_ghost_fluid.hpp ------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  riemann_ghost_fluid.hpp
/// \brief This file defines an implementation of the riemann ghost fluid. For
///        details of the specific implementation, refer to: ...
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_RIEMANN_GHOST_FLUID_HPP
#define FLUIDITY_GHOST_FLUID_RIEMANN_GHOST_FLUID_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {


/// This struct implements the ghost fluid method, which takes a tuple of 
/// material iterator wrappers and loads ghost cell values into the state data
/// for a material using the levelsets for the materials.
///
/// This is the simplest method of setting the ghost cell data, and uses the
/// method of Fedkiw, et al (A Non-Oscillatory Eularian Approach to Interfaces
/// in Multi-Material Flows (The Ghost Fluid Method)) to set the ghost cells for
/// for material A by copying the pressure and velocity from the real cells in
/// material B and extrapolating the entropy from the real cells in material A
/// to the ghost cells for material A.
template <std::size_t Width>
struct RiemannGFM {

};

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_RIEMANN_GHOST_FLUID_HPP

