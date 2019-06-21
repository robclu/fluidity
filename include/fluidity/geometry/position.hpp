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

/// The Pos class represents a position in 3D space.
/// \tparam T The type of the position data.
template <typename T>
class Pos : public Array<T, 3> {
  /// Defines the type of the base storage type.
  using base_t = Array<T, 3>;

 public:
  /// Inherit all constructors.
  using base_t::base_t;

  /// Returns the x-coord of the postition.
  fluidity_host_device constexpr auto x() const -> T {
    return base_t::operator[](0);
  }

  /// Returns the y-coord of the postition.
  fluidity_host_device constexpr auto y() const -> T {
    return base_t::operator[](1);
  }

  /// Returns the x-coord of the postition.
  fluidity_host_device constexpr auto z() const -> T {
    return base_t::operator[](2);
  }
};

}} // namespace fluid::geometry

#endif // FLUIDITY_GEOMETRY_POSITION_HPP
