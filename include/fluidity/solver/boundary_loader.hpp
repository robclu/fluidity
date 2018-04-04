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
  /// \param[in]  dim   The dimension to configure the type of boundary setting
  ///                   for.
  /// \param[in]  index The index of the boundary to set (first or second)
  /// \param[in]  type  The type to set the boundary to.
  /// \tparam     Value The value which defines the dimension.
  template <std::size_t Value>
  fluidity_host_device constexpr void
  configure(Dimension<Value> /*dim*/, BoundaryIndex index, BoundaryKind kind)
  {
    _configuration ^= (-static_cast<uint8_t>(kind) ^ _configuration) // Value
                    & (1 << bit_index(Dimension<Value>{}, index));
  }

  /// Overload of operator() to set the value of the \p boundary state to that
  /// of the \p internal state. If the configured type of the boundary for the
  /// \p index boundary in the given dimension is BoundaryKind::reflective, then
  /// the velocity normal to the dimension is flipped.
  /// \param[in] internal The internal state to use to set the boundary state.
  /// \param[in] boundary The boundary state to load.
  /// \param[in] index    The index of the boundary (first or second).
  /// \tparam    State    The type of the state.
  /// \tparam    Value    The value which defines the dimension.
  template <typename State, std::size_t Value>
  fluidity_host_device void operator()(State&&          internal,
                                       State&&          boundary,
                                       Dimension<Value> /*dim*/ ,
                                       BoundaryIndex    index   ) const
  {
    boundary = internal;
    set_velocity(std::forward<State>(boundary), Dimension<Value>{}, index);
  }

  /// Sets the velocity of the \p boundary by flipping it's value if the kind
  /// of the boundary is BoundaryKind::reflective, otherwise nothing is done.
  /// \param[in] boundary The boundary state to set the velocity for.
  /// \param[in] index    The index of the boundary (first or second).
  /// \tparam    State    The type of the state.
  /// \tparam    Value    The value which defines the dimension.
  template <typename State, std::size_t Value>
  fluidity_host_device void set_velocity(State&&          boundary,
                                         Dimension<Value> /*dim*/ ,
                                         BoundaryIndex    index   ) const
  {
    // Simply need to change the velocity for dimension __dim__. 
    if (get_kind(Dimension<Value>{}, index) == BoundaryKind::reflective)
    {
      constexpr auto dim = Dimension<Value>{};
      boundary.set_velocity(-boundary.velocity(dim), dim);
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
  /// \param[in]  elements The number of elements in the data for the dim.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     Iterator The type of the iterators.
  /// \tparam     Value    The value which defines the dimension.
  template < typename    Iterator
           , std::size_t Value
           , std::enable_if_t<
               exec::is_gpu_policy_v<typename Iterator::exec_policy_t>, int> = 0
           >>
  static fluidity_host_device void
  load_boundary(Iterator&&            data     ,
                Iterator&&            patch    ,
                std::size_t           elements ,
                Dimension<Value>      /*dim*/  ,
                const BoundarySetter& setter   )
  {
    constexpr auto dim = Dimension<Value>{};

    // The global index is used to load additional data at the boundaries of
    // the domain, and the local index is used to load extra data which is
    // inside the domain.
    auto shift = global_id(dim)
    if (shift < padding)
    {
      const auto elem_to_use = *data.offset(2 * shift + 1  , dim);
      const auto elem_to_set = *patch.offset(-2 * shift - 1, dim);
      setter(elem_to_use, elem_to_set, dim, BoundaryIndex::first);
    }

    // Move id values to end end of the block:
    shift = elements - global_id(dim) - 1;
    if (shift < padding)
    {
      const auto elem_to_use = *data.offset(elements - 2 * shift - 1, dim);
      const auto elem_to_set = *patch.offset(2 * shift + 1, dim);
      setter(elem_to_use, elem_to_set, dim, BoundaryIndex::second);
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
  /// \param[in]  elements The number of elements in the data for the dim.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     Iterator The type of the iterators.
  /// \tparam     Value    The value which defines the dimension.
  template < typename    Iterator
           , std::size_t Value
           , std::enable_if_t<
               exec::is_cpu_policy_v<typename Iterator::exec_policy_t>, int> = 0
           >>
  static fluidity_host_device void
  load_boundary(Iterator&&            data     ,
                Iterator&&            patch    ,
                std::size_t           elements ,
                Dimension<Value>      /*dim*/  ,
                const BoundarySetter& setter   )
  {
    constexpr auto dim = Dimension<Value>{};

    auto front = data + padding;
    auto back  = data + elements - padding;

    unrolled_for<padding>([&] (auto i)
    {

    });
  }

  /// This sets the padding elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// p = padding element to set in the patch
  /// s = data element to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | p-n | p-1 | p0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | p0 | p1 | pn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// This implementation is only enabled when the iterator is a GPU iterator.
  /// 
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  elements The number of elements in the patch.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \tparam     Iterator The type of the iterators.
  /// \tparam     Value    The value which defines the dimension.
  template < typename    Iterator
           , std::size_t Value
           , std::enable_if_t<
               exec::is_gpu_policy_v<typename Iterator::exec_policy_t>, int> = 0
           >>
  static fluidity_host_device void
  load_patch(Iterator&& patch, std::size_t elements, Dimension<Value> /*dim*/)
  {
    constexpr auto dim = Dimension<Value>{};
    if (thread_id(dim) < padding)
    {
      *data = *data.offset(-2 * thread_id(dim) - 1, dim);
    }
    else if (thread_id(dim) > elements - padding)
    {
      *data = *data.offset(2 * thread_id(dim) + 1, dim);
    }
  }

  /// This sets the padding elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// p = padding element to set in the patch
  /// s = data element to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | p-n | p-1 | p0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | p0 | p1 | pn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// This implementation is only enabled when the iterator is a CPU iterator.
  /// 
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  elements The number of elements in the patch.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \tparam     Iterator The type of the iterators.
  /// \tparam     Value    The value which defines the dimension.
  template < typename    Iterator
           , typename    Data
           , std::size_t Value
           , std::enable_if_t<
               exec::is_cpu_policy_v<typename Iterator::exec_policy_t>, int> = 0
           >>
  static fluidity_host_device void load(Iterator&&       data     ,
                                        std::size_t      elements ,
                                        Dimension<Value> /*dim*/  )
  {
    auto front = data + padding;
    auto back  = data + elements - padding;

    unrolled_for<padding>([&] (auto i)
    {
      constexpr auto dim   = Dimension<Value>{};
      constexpr auto shift = -2 * i - 1;

      *front                   = *front.offset(shift, dim);
      *back.offset(shift, dim) = *back;

      front++: back++;
    });
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP