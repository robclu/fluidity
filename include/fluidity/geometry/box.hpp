//==--- fluidity/geometry/box.hpp -------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  box.hpp
/// \brief This file defines a clas to represent a box in 2 or 3 dimensions.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GEOMETRY_BOX_HPP
#define FLUIDITY_GEOMETRY_BOX_HPP

#include "position.hpp"
#include <fluidity/container/vec.hpp>
#include <fluidity/math/math.hpp>

namespace fluid    {
namespace geometry {

/// Represents a box in 3 dimensions, or a square in 2 dimensions.
/// \tparam T The type of the data for the box.
template <typename T>
class Box {
 public:
  /// Defines the type of the position data.
  using pos_t    = Pos<T>;
  /// Defines the type for the sides of the box.
  using length_t = Vec3<T>;

  /// Default constructor which creates a point centered at the origin.
//fluidity_host_device constexpr Box() = default;

  /// Creates a box specifying the lengths of each side.
  /// \param[in] lengths The lengths of each side of the box.
  fluidity_host_device constexpr Box(const length_t& lengths)
  : _sides(lengths) {}

  /// Creates a box specifying the lengths of each side, with the center of the
  /// box at \p p.
  /// \param[in] lengths The lengths of each side of the box.
  /// \param[in] p       The position of the center of the box.
  //fluidity_host_device Box(const sides_t& lengths, const pos_t& p)
  fluidity_host_device Box(const length_t& lengths, const pos_t& p)
  : _sides(lengths), _center(p) {}

  /// Returns the distance of a point at position \p pos to the sphere. A
  /// positive result is returned for values outside the sphere.
  /// \param[in] pos The position to get the distance to.
  fluidity_host_device auto distance(const pos_t& pos) const -> T {
    using namespace math;
    // Move back to the origin, and compute dist to each side.
    const auto dist = abs(pos - _center) - (_sides / T(2));
    return length(max(dist, T(0.0)))
      + min(max(dist[0], max(dist[1], dist[2])), T(0.0));
  }

 private:
  length_t _sides  = {1, 1, 1};      //!< The position of the center of the sphere.
  pos_t    _center = {0, 0, 0};  //!< The center of the box.
};

}} // namespace fluid::geometry

#endif // FLUIDITY_GEOMETRY_BOX_HPP
