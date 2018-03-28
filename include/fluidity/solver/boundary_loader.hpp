//==--- fluidity/solver/boundary_loader.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  boundary_loader.hpp
/// \brief This file defines functionality for loading data at the boundary.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP
#define FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP

namespace fluid  {
namespace solver {

/// The BoundaryKind enum defines the kind of a boundary.
enum class BoundaryKind {
  transmissive = 0,   //!< Defines a transmissive boundary.
  reflective   = 1    //!< Defines a reflective boundary.   
};

/// The BoundaryIndex enum defines which of the boundaries to set.
enum class BoundaryIndex {
  First  = 0, //!< Defines the index of the first boundary.
  Second = 1  //!< Defines the index of the second boundary.
};

/// The BoundaryLoader class implements functionality for loading data at the
/// boundary.
struct BoundaryLoader {
  /// Defines the mask to get a BoundaryKind from an integer.
  static constexpr uint16_t mask = 0x0001;

  /// Configures the \p ith boundary for \p dim.
  /// \param[in]  dim   The dimension to configure the type of boundary setting
  ///                   for.
  /// \param[in]  index The index of the boundary to set (first or second)
  /// \param[in]  type  The type to set the boundary to.
  fluidity_host_device constexpr void
  configure(Dimension<Value> /*dim*/, BooundayIndex index, BoundaryKind kind)
  {
    _configuration ^= (-static_cast<uint8_t>(kind) ^ _configuration) // Value
                    & (1 << bit_index(Dimension<Value>{}, index));
  }

  /// Overload of operator() to set the state of the \p boundary state to that
  /// of the \p internal state. If the configured type of the boundary for the
  /// \p index boundary in dimension \p dim is Type::Reflective, then the
  /// velocity normal to \p dim is flipped.
  template <typename State, std::size_t Value>
  fluidity_host_device void operator()(State&&          internal,
                                       State&&          boundary,
                                       Dimension<Value> /*dim*/ ,
                                       BoundaryIndex    index   ) const
  {
    *boundary = *internal;
    set_velocity(std::forward<State>(boundary), Dimension<Value>{}, index);
  }

  /// Overload of operator() to set the state of the \p boundary state to that
  /// of the \p internal state. If the configured type of the boundary for the
  /// \p index boundary in dimension \p dim is Type::Reflective, then the
  /// velocity normal to \p dim is flipped.
  /// 
  /// This overload allows the index to be specified as a numeric type, which
  /// allows the index to be set more genericaly.
  template <typename State, std::size_t Value>
  fluidity_host_device void set_velocity(State&&          boundary,
                                         Dimension<Value> /*dim*/ ,
                                         BoundaryIndex    index   ) const
  {
    // Simply need to change the velocity for dimension __dim__. 
    if (get_kind(Dimension<Value>{}, index) == BoundarKind::reflective)
    {
      constexpr auto dim = Dimension<Value>{};
      boundary.velocity(dim) = -boundary.velocity(dim);
    }
  }

 private:
  uint16_t _configuration = 0;  //!< Boundary configuration -- all transmissive.
  
  /// Returns the shift amount to move the bits for the boundary type for
  /// dimension \p dim and \p index into the LSB position.
  /// \param[in]  dim     The dimension to get the bit index for.
  /// \param[in]  index   The index (0 = front, 1 = back) of the boundary type
  ///                     to get.
  template <std::size_t Value>
  fluidity_host_device constexpr std::size_t
  bit_index(Dimension<Value> /*dim*/, BoundaryIndex index) const
  {
    return (static_cast<std::size_t>(Value) << 2) + 
           (static_cast<std::size_t>(index) << 1);
  }

  /// Returns the type of the boundary for dimension \p with index \p index.
  /// \param[in]  dim     The dimension to get the boundary type for.
  /// \param[in]  index   The index (0 = front, 1 = back) of the boundary type
  ///                     to get.
  template <std::size_t Value>
  fluidity_host_device constexpr BoundaryKind
  get_kind(Dimension<Value> /*dim*/, BoundaryIndex index) const
  {
    return
      static_cast<BoundaryKind>(
        (_configuration >> bit_index(Dimension<Value>{}, index)) & mask
      );
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP