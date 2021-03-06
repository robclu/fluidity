//==--- fluidity/container/position.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  position.hpp
/// \brief This file defines a to represent a position in 3d space.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GEOMETRY_POSITION_HPP
#define FLUIDITY_GEOMETRY_POSITION_HPP

#include <fluidity/container/array.hpp>

namespace fluid    {
namespace geometry {

/// Defines an alias for a position type for 2 dimensions. Currently this is
/// just an alias to a 2D array.
/// \tparam T The type of the position data.
template <typename T>
using Pos2 = Array<T, 2>;

/// Defines an alias for a position type for 3 dimensions. Currently this is
/// just an alias to a 3D array.
/// \tparam T The type of the position data.
template <typename T>
using Pos3 = Array<T, 3>;

}} // namespace fluid::geometry

#endif // FLUIDITY_GEOMETRY_POSITION_HPP
