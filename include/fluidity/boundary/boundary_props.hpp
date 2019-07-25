//==--- fluidity/boundary/boundary_props.hpp --------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  boundary_props.hpp
/// \brief This file defines properties of boundaries.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_BOUNDARY_BOUNDARY_PROPS_HPP
#define FLUIDITY_BOUNDARY_BOUNDARY_PROPS_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace boundary {

/// The BoundaryKind enum defines the kind of a boundary.
enum class BoundaryKind : uint8_t {
  transmissive = 0,   //!< Defines a transmissive boundary.
  reflective   = 1    //!< Defines a reflective boundary.   
};

/// The DomainBoundary class represents information for the boundaries for a
/// specific dimension of a domain, where there is a boundary on each side of
/// the domain.
///
/// The boundary also contains information about where the end of the domain is,
/// and hence where the boundary is located.
class DomainBoundary {
 public:
  /// Default constructor, creates a transmissive boundary, at the beginning of
  /// the domain.
  fluidity_host_device constexpr DomainBoundary()
  : _end_pos{0}                       ,
    _start(BoundaryKind::transmissive),
    _end(BoundaryKind::transmissive)  {}

  /// Initializes the type of the boundary on each side of dimension for the
  /// domain.
  fluidity_host_device constexpr
  DomainBoundary(BoundaryKind start, BoundaryKind end, std::size_t end_pos)
  : _end_pos(end_pos), _start(start), _end(end) {}

  /// Returns the kind of the boundary at the start of the domain.
  fluidity_host_device constexpr auto start() const -> BoundaryKind {
    return _start;
  }

  /// Returns the kind of the boundary at the end of the domain.
  fluidity_host_device constexpr auto end() const -> BoundaryKind {
    return _end;
  }

  /// The position of the end of the boundary in the domain.
  fluidity_host_device constexpr auto end_position() const -> std::size_t {
    return _end_pos;
  }

 private:
  std::size_t  _end_pos = 0;  //!< The position of the boundary in the domain.
  BoundaryKind _start;        //!< The kind of the start boundary.
  BoundaryKind _end;          //!< The kind of the end boundary.
};

}} // namespace fluid::boundary

#endif // FLUIDITY_BOUNDARY_BOUNDARY_PROPS_HPP
