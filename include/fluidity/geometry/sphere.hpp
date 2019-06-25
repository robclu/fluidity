//==--- fluidity/geometry/sphere.hpp ----------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  sphere.hpp
/// \brief This file defines a class to represent a sphere.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GEOMETRY_SPHERE_HPP
#define FLUIDITY_GEOMETRY_SPHERE_HPP

#include "position.hpp"
#include <fluidity/math/math.hpp>

namespace fluid    {
namespace geometry {

/// Represents a sphere.
/// \tparam T The type of the data for the sphere.
template <typename T>
class Sphere {
 public:
  /// Defines the type of the position data.
  using pos_t = Pos<T>;

  /// Default constructor which creates a point centered at the origin.
  fluidity_host_device constexpr Sphere() = default;

  /// Creates a sphere, centered at the origin, with a radius of \p r.
  /// \param[in] r The radius of the sphere.
  fluidity_host_device constexpr Sphere(T r) : _r(r) {}

  /// Create s sphere, centered at x, y, z, with a radius of 0 i.e a point.
  /// \param[in] x The x position of the sphere center.
  /// \param[in] y The y position of the sphere center.
  /// \param[in] z The z position of the sphere center.
  fluidity_host_device constexpr Sphere(const pos_t& pos)
  : _pos{pos} {}

  /// Create s sphere, centered at x, y, z, with a radius of r.
  /// \param[in] x The x position of the sphere center.
  /// \param[in] y The y position of the sphere center.
  /// \param[in] z The z position of the sphere center.
  /// \param[in] r The radius of the sphere.
  fluidity_host_device constexpr Sphere(const pos_t& pos, T r)
  : _pos{pos}, _r(r) {}

  /// Returns the distance of a point at position \p pos to the sphere. A
  /// positive result is returned for values outside the sphere.
  /// \param[in] pos The position to get the distance to.
  fluidity_host_device auto distance(const pos_t& pos) const -> T {
    const auto d = math::length(pos - _pos);
    return d - _r;
  }

 private:
  pos_t _pos = {0, 0, 0}; //!< The position of the center of the sphere.
  T     _r   = 0;         //!< The radius of the sphere.
};

/// Creates a new sphere as a circle, centered at x, y, with radius r.
/// \param[in] x The x position of the circle center.
/// \param[in] y The y position of the circle center.
/// \param[in] r The radius of the circle. 
/// \tparam    T The data type for the circle.
template <typename T>
fluidity_host_device constexpr auto circle(T x, T y, T r = T(1)) -> Sphere<T> {
  using sphere_t = Sphere<T>;
  using pos_t    = typename sphere_t::pos_t;
  return Sphere<T>(pos_t{x, y, T(0)}, r);
}

}} // namespace fluid::geometry

#endif // FLUIDITY_GEOMETRY_SPHERE_HPP

