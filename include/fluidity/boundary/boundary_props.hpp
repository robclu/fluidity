//==--- fluidity/boundary/boundary_kind.hpp ---------------- -*- C++ -*- ---==//
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

#ifndef FLUIDITY_SOLVER_BOUNDARY_PROPS_HPP
#define FLUIDITY_SOLVER_BOUNDARY_PROPS_HPP

namespace fluid {

/// The BoundaryKind enum defines the kind of a boundary.
enum class BoundaryKind {
  transmissive = 0,   //!< Defines a transmissive boundary.
  reflective   = 1    //!< Defines a reflective boundary.   
};

/// The DomainBoundary class represents information for the boundaries for a
/// specific dimension of a domain, where there is a boundary on each side of
/// the domain.
class DomainBoundary {
 public:
  /// Initializes the type of the boundary on each side of dimension for the
  /// domain.
  fuidity_host_device DomainBoundary(BoundaryKind start, BoundaryKind end)
  : _start(start), _end(end) {}


  /// Returns the kind of the boundary at the start of the domain.
  fluidity_host_device constexpr auto start() -> BoundaryKind {
    return _start;
  }

  fluidity_host_device constexpr auto end() -> BoundaryKind {
    _end;
  }
  `
 private:
  BoundaryKind _start; //!< The kind of the boundary at the start of the domain,
  BoundaryKind _end;   //!< The kind of the boundary at the end of the domain.
}

} // namespace fluid

#endif // FLUIDITY_SOLVER_BOUNDARY_PROPS_HPP
