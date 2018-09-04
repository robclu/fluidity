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

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/utility/cuda.hpp>

namespace fluid  {
namespace solver {

/// The BoundaryKind enum defines the kind of a boundary.
enum class BoundaryKind {
  transmissive = 0,   //!< Defines a transmissive boundary.
  reflective   = 1    //!< Defines a reflective boundary.   
};

/// The BoundaryIndex enum defines which of the boundaries to set.
enum class BoundaryIndex {
  first  = 0, //!< Defines the index of the first boundary.
  second = 1  //!< Defines the index of the second boundary.
};

/// The BoundarySetter class implements functionality for setting boundary data
/// based on an internal state.
struct BoundarySetter {
  /// Defines the mask to get a BoundaryKind from an integer.
  static constexpr uint16_t mask = 0x0001;

  /// Configures the \p ith boundary for \p dim.
  /// \param[in]  dim   The dimension to configure the boundary setting for.
  /// \param[in]  index The index of the boundary to set (first or second)
  /// \param[in]  kind  The kind of the boundary.
  /// \param[in]  type  The type to set the boundary to.
  /// \tparam     Dim   The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr void
  configure(Dim dim, BoundaryIndex index, BoundaryKind kind)
  {
    _configuration ^= (-static_cast<uint8_t>(kind) ^ _configuration)
                    & (1 << bit_index(dim, index));
  }

  /// Overload of operator() to set the value of the \p boundary state to that
  /// of the \p internal state. If the configured type of the boundary for the
  /// \p index boundary in the given dimension is BoundaryKind::reflective, then
  /// the velocity normal to the dimension is flipped.
  /// \param[in] internal   The internal state to use to set the boundary state.
  /// \param[in] boundary   The boundary state to load.
  /// \param[in] dim        The dimension to set the value in.
  /// \param[in] index      The index of the boundary (first or second).
  /// \tparam    StateType  The type of the state.
  /// \tparam    Dim        The type of the dimension.
  template <typename StateType, typename Dim>
  fluidity_host_device void operator()(const StateType& internal,
                                       StateType&       boundary,
                                       Dim              dim     ,
                                       BoundaryIndex    index   ) const
  {
    boundary = internal;
    set_velocity(std::forward<StateType>(boundary), dim, index);
  }

  /// Sets the velocity of the \p boundary by flipping it's value if the kind
  /// of the boundary is BoundaryKind::reflective, otherwise nothing is done.
  /// \param[in] boundary   The boundary state to set the velocity for.
  /// \param[in] dim        The dimension to set the velocity for.
  /// \param[in] index      The index of the boundary (first or second).
  /// \tparam    StateType  The type of the state.
  /// \tparam    Dim        The type of the dimension.
  template <typename StateType, typename Dim>
  fluidity_host_device void
  set_velocity(StateType&& boundary, Dim dim, BoundaryIndex index) const
  {
    if (get_kind(dim, index) == BoundaryKind::reflective)
    {
      boundary.set_velocity(-boundary.velocity(dim), dim);
    }
  }

 private:
  uint16_t _configuration = 0;  //!< Boundary configuration -- all transmissive.
  
  /// Returns the shift amount to move the bits for the boundary type for
  /// dimension \p dim and \p index into the LSB position.
  /// \param[in]  dim     The dimension to get the bit index for.
  /// \param[in]  index   The index (0 = front, 1 = back) of the boundary.
  /// \tparam     Dim     The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr std::size_t
  bit_index(Dim dim, BoundaryIndex index) const
  {
    return (static_cast<std::size_t>(dim) << 2) + 
           (static_cast<std::size_t>(dim) << 1);
  }

  /// Returns the type of the boundary for dimension \p with index \p index.
  /// \param[in]  dim     The dimension to get the boundary type for.
  /// \param[in]  index   The index (0 = front, 1 = back) of the boundary.
  /// \tparam     Dim     The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr BoundaryKind
  get_kind(Dim dim, BoundaryIndex index) const
  {
    return static_cast<BoundaryKind>(
             (_configuration >> bit_index(dim, index)) & mask);
  }
};

/// The BoundaryLoader struct loads state data into "extra" cells required by a
/// computational block, i.e the additional states required by a cell which is
/// at the start or end of a computational block. For blocks at the start and
/// end of a dimensions the "extra" cells need to be set based on the type of
/// boundary in the computational domain, while for other blocks (i.e if the
/// domain is split into patches) then the values comes from inside the domain.
/// 
/// The loader functionality is thus split into boundary loading and internal
/// loading functionality.
///
/// \tparam Padding The width of the padding (per edge).
template <std::size_t Padding>
struct BoundaryLoader {
  /// Defines the amount of padding used by the loader.
  static constexpr std::size_t padding = Padding;

  /// This sets the boundary elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// b = boundary element to set in the patch
  /// s = valid global data to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | b-n | b-1 | b0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | b0 | b1 | bn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// With the above illustration, the technique used in the code should be
  /// relatively simple to understand.
  /// 
  /// This implementation is only enabled when the iterator is a GPU iterator.
  /// 
  /// \param[in]  data     An iterator to global data.
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     I1       The type of the data iterator.
  /// \tparam     I2       The type of the patch iterator.
  /// \tparam     Dim      The type of the dimension.
  template < typename I1, typename I2, typename Dim, exec::gpu_enable_t<I1> = 0>
  static fluidity_host_device void
  load_boundary(I1&& data, I2&& patch, Dim dim, BoundarySetter setter)
  {
    int global_idx = flattened_id(dim), local_idx = thread_id(dim);
    if (global_idx < padding)
    {
      constexpr auto bi = BoundaryIndex::first;
      setter(*patch, *patch.offset(-2 * global_idx - 1, dim), dim, bi);
    }
    else if (local_idx < padding)
    {
      const auto shift = -2 * local_idx - 1;
      *patch.offset(shift, dim) = *data.offset(shift, dim);
    }

    global_idx = static_cast<int>(data.size(dim)) - global_idx - 1;
    local_idx  = static_cast<int>(block_size(dim)) - local_idx - 1;

    if (global_idx < padding)
    {
      constexpr auto bi = BoundaryIndex::second;
      setter(*patch, *patch.offset(2 * global_idx + 1, dim), dim, bi);
    }
    else if (local_idx < padding)
    {
      const auto shift = 2 * local_idx + 1;
      *patch.offset(shift, dim) = *data.offset(shift, dim);
    }
  }

  /// This sets the boundary elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// b = boundary element to set in the patch
  /// s = valid global data to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | b-n | b-1 | b0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | b0 | b1 | bn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// With the above illustration, the technique used in the code should be
  /// relatively simple to understand.
  /// 
  /// This implementation is only enabled when the iterator is a CPU iterator.
  /// 
  /// \param[in]  data     An iterator to global data.
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     I1       The type of the data iterator.
  /// \tparam     I2       The type of the patch iterator.
  /// \tparam     Dim      The type of the dimension.
  template < typename I1, typename I2, typename Dim, exec::cpu_enable_t<I1> = 0>
  static fluidity_host_device void
  load_boundary(I1&& data, I2&& patch, Dim dim, BoundarySetter setter)
  {
    /*
     * TODO: Implement ...
     */
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP