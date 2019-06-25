//==--- fluidity/geometry/triangle.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  triangle.hpp
/// \brief This file defines a clas to represent a triangle.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GEOMETRY_TRIANGLE_HPP
#define FLUIDITY_GEOMETRY_TRIANGLE_HPP

#include "position.hpp"
#include <fluidity/container/vec.hpp>
#include <fluidity/math/math.hpp>

namespace fluid    {
namespace geometry {

/// Represents a triangle.
/// \tparam T The type of the data for the triangle.
template <typename T>
class Triangle {
 public:
  /// Defines the type of the position data.
  using pos2_t = Pos2<T>;

  /// Creates a box specifying the lengths of each side.
  /// \param[in] lengths The lengths of each side of the box.
  fluidity_host_device constexpr Triangle(pos2_t p0, pos2_t p1, pos2_t p2)
  : _p0(p0), _p1(p1), _p2(p2) {}

  /// Returns the distance of a point at position \p pos to the sphere. A
  /// positive result is returned for values outside the sphere.
  /// \param[in] pos The position to get the distance to.
  fluidity_host_device auto distance(const pos2_t& pos) const -> T {
    using namespace math;

    auto e0 = _p1 - _p0, e1 = _p2 - _p1, e2 = _p0 - _p2;
    auto v0 = pos - _p0, v1 = pos - _p1, v2 = pos - _p2;

    auto pq0 = v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
    auto pq1 = v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
    auto pq2 = v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);
                        
    auto s = signum(e0[0] * e2[1] - e0[1] * e2[0]);
    auto d = min(
      min(
        pos2_t(dot(pq0, pq0), s * (v0[0] * e0[1] - v0[1] * e0[0])),
        pos2_t(dot(pq1, pq1), s * (v1[0] * e1[1] - v1[1] * e1[0]))
      ),
      pos2_t(dot(pq2, pq2), s * (v2[0] * e2[1] - v2[1] * e2[0]))
    );
    return -sqrt(d[0]) * signum(d[1]);
  }

 private:
  pos2_t _p0 = 0.0; //!< The position of first vertex.
  pos2_t _p1 = 0.0; //!< The position of second vertex.
  pos2_t _p2 = 0.0; //!< The position of third vertex.
};

}} // namespace fluid::geometry

#endif // FLUIDITY_GEOMETRY_TRIANGLE_HPP
