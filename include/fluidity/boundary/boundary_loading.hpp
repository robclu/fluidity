//==--- fluidity/boundary/boundary_loader.hpp -------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  boundary_loader.hpp
/// \brief This file defines functionality for loading data at the boundary and
///        into shared memory padding.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_BOUNDARY_BOUNDARY_LOADER_HPP
#define FLUIDITY_boundary_BOUNDARY_LOADER_HPP

#include <fluidity/state/state_traits.hpp>

namespace fluid    {
namespace boundary {
namespace detail {

/// Returns the shift amount for an iterator when loading boundary data. This
/// computation assumes that the Nth thread (or Nth from the end) in a block
/// must load the -Nth (end + [Nth from end]) padding cell. This also assumes
/// that the iters have been offset in each dimension such that the \p it_to
/// and \p it_from iterators point to the data cell corresponding to their index
/// in the data grid, i.e the iterator for thread idx {0, 0, 0} points to the
/// cell for that data. For example, for dim 0, and a 5 thread grid with 2
/// padding cells, the following would happen:
///
/// \code{bash}
///
/// Thread:     -     -     0     1     2     3     4     -     -
/// Cell  :     p     p     0     1     2     3     4     p     p
/// Load  : _________________________________________________________
///                   -------           x           -------
///                   |     |           x           |     |
///                   V     |           x           |     V
///           =======================================================
///           |     |     |     |     |     |     |     |     |     |
///           =======================================================
///             ^                  |    x     |                  ^
///             |                  |    x     |                  |
///             --------------------    x     --------------------
///         __________________________________________________________
/// \endcode
/// \param[in] index The index of the iterator in the block
template <typename T>
fluidity_host_device auto shift(T amount, T sign) -> T {
  return T(2) * sign * amount  + sign;
}

/// Sets the boundary using the \p data_from data to set the \p data_to data,
/// and the \p kind of the boundary. This overload is only enabled when the type
/// of the \p data_from is a state.
/// \param[in] data_from The data to use to set the to data.
/// \param[in] data_to   The data to set.
/// \param[in] kind      The kind of the boundary.
/// \tparam    DataFrom  The type of the from data.
/// \tparam    DataTo    The type of the to data.
template <typename DataFrom,
          typename DataTo  ,
          typename Dim     , state_enable_t<DataFrom> = 0>
fluidity_host_device void set_boundary(const DataFrom& data_from,
                                       const DataTo&   data_to  ,
                                       BoundaryKind    kind     ,
                                       Dim             dim      ) {
  data_to = data_from;
  if (kind == BoundaryKind::reflective) {
    data_to.set_velocity(-data_from.velocity(dim), dim)
  } 
}

/// Sets the boundary using the \p data_from data to set the \p data_to data,
/// and the \p kind of the boundary. This overload is only enabled when the type
/// of the \p data_from is not a state.
/// \param[in] data_from The data to use to set the to data.
/// \param[in] data_to   The data to set.
/// \param[in] kind      The kind of the boundary.
/// \tparam    DataFrom  The type of the from data.
/// \tparam    DataTo    The type of the to data.
template <typename DataFrom, typename DataTo, non_state_enable_t<DataFrom> = 0>
fluidity_host_device void set_boundary(const DataFrom& data_from,
                                       const DataTo&   data_to  ,
                                       BoundaryKind    kind     ,
                                       Dim             dim      ) {
  data_to = data_from;
  if (kind = BoundaryKind::reflective) {
    data_to = -data_to;
  }
}

} // namespace detail

/// This loads the padding data for the \p it_to data. This uses the \p it_from
/// iterator and the boundary conditions specified by the \p boundaties for each
/// dimension, to set the data at if the thread index is at the start or the end
/// of the domain. 
///
/// For threads which are at the start of the block, the \p it_from data is
/// copied into the padding cell at the computed offset amount from \p it_to
/// and \p it_from iterators.
///
/// Here, the \p it_from iterator should be an iterator over the global data,
/// which has been offset in each dimension by the global thread index, while
/// the \p it_to iterator should be an iterator over the shared memory data,
/// where the iterator has been offset in each dimension by the thread index in
/// that dimension.
///
/// \param[in] it_from    The iterator to get the data to set from.
/// \param[in] it_to      The iterator to set the padding for.
/// \param[in] boundaries The information for the boundaries. 
/// \tparam    W          The width of the padding.
/// \tparam    ItFrom     The type of the from iterator.
/// \tparam    ItTo       The type of the to iterator.
/// \tparam    Boundaries The type of the boundary information.
template <std::size_t W, typename ItFrom, typename ItTo, typename Boundaries>
fluidity_host_device void 
load_padding_with_boundaries(ItFrom&&     it_from   ,
                             ItTo&&       it_to     ,
                             Boundaries&& boundaries) {
  constexpr auto dims  = std::decay_t<ItFrom>::dimensions;
  constexpr auto width = W;
  
  unrolled_for<dims>([&] (auto dim) {
    auto bound = boundaries[dim];
    auto idx   = std::min(flattened_id(dim), bound.size() - width);
    auto sign  = math::signum(flattened_id(dim) - 0.5 * bound.size());
    if (idx < width) {
      detail::set_boundary(
        it_from,
        it_to.offset(detail::shift(idx, sign), dim),
        sign > 0 ? bound.end() : bound.start(),
        dim
      );
      break;
    }

    // Not a global boundary, try block boundary.
    idx  = std::min(thread_id(dim), block_size(dim) - width);
    sign = math::signum(thread_id(dim) - 0.5 * block_size(dim));
    if (idx < width) {
      const auto amount = detail::shift(idx, sign);
      *it_to.offset(amount, dim) = *it_from.offset(amount, dim);
    }
  });
}

}} // namespace fluid::boundary 

#endif // FLUIDITY_boundary_BOUNDARY_LOADER_HPP
