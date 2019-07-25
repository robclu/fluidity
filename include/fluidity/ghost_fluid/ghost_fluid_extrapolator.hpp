//==--- fluidity/ghost_fluid/ghost_cell_extrapolator.hpp --- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ghost_cell_extrapolator.hpp
/// \brief This file defines a class which extrapolates star state data away
///        from the interface into the ghost cells.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_GHOST_CELL_EXTRAPOLATOR_HPP
#define FLUIDITY_GHOST_FLUID_GHOST_CELL_EXTRAPOLATOR_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {

/// This class extrapolates data from the interfacial cells into the ghost
/// cells, using a simple upwinding procedure to determine which cells to use to
/// set the data. For this to function correctly as the GFM, the interfacial
/// cells should be set to the star state values computed by solving the Riemann
/// problem normal to the inteface, at each interfacial cell, and then rotated
/// back into the global co-ordinate frame. This can by done by either calling
/// the `...` method on an entire grid, or by invoking the `...` method on the
/// cell iterator.
///
/// This implementation is essentially the same implementation that is used by
/// the fast sweeping method, but rather than extrapolating the levelset data,
/// the state data is extrapolated based on the levelset data.
///
/// TODO: Add further explanation for how this works.
struct GhostCellExtrapolator {
  /// Defines the width of the extrapolator, which is the number of neighbouring
  /// cells that are needed when setting a single cell. Note that this is not
  /// the number of ghost cells which are set, which is provied at runtime.
  static constexpr auto width = 1;

  /// Invokes the extrapolation on the material iterators.
  /// \param[in] mat_iterator  The type of the material iterator.
  /// \param[in] dh            The resolution of the material data grids.
  /// \tparam    MatIterator   The type of the material iterator.
  /// \tparam    T             The type of the resolution data.
  template <typename MatIterator, typename T>
  fluidity_host_device static void
  invoke(MatIterator&& mat_iterator, T dh, std::size_t band_width = 3) {
    static_assert(traits::is_mat_iterator_v<MatIterator>,
      "Extrapolation of ghost cells requires a material iterator!");
    using mat_iter_t   = std::decay_t<MatIterator>;
    using state_data_t = std::decay_t<decltype(*mat_iter.state_it)>;

    // The amount of padding for the loading. We need on element on each side
    // for each of the dimensions.
    constexpr auto pad  = std::size_t{1};
    constexpr auto dims = mat_iter_t::dimensions;

    // Shared memory iterators for the state and levelset data. Here we only
    // need a single padding element on each side.
    auto ls_it    = make_multi_iter<1>(mat_iter.ls_it);
    auto state_it = make_multi_iter<1>(mat_iter.state_it);

    // Offset the iterators ...
    unrollled_for<dims>([&] (auto dim) {
      const auto offset = thread_id(dim) + 1;
      mat_iter.shift(flattened_id(dim), dim);
      ls_it.shift(offset, dim);
      state_it.shift(offset, dim);
    });

    // Set the boundary data ...
    unrolled_for<iter_t::dimensions>([&] (auto dim) {
      if (flattened_id(dim) == 0) {
        *ls_it.offset(-1, dim) = std::numeric_limits<value_t>::max(); 
      } else if (flattened_id(dim) >= output.size(dim) - 1) {
        *ls_it.offset(1, dim) = std::numeric_limits<value_t>::max(); 
      } else if (thread_id(dim) == 0) {
        *state_it.offset(-1, dim) = *mat_iter.state_it.offset(-1, dim);
        *ls_it.offset(-1, dim)    = *mat_iter.ls_it.offset(-1, dim);
      } else if (thread_id(dim) == block_size(dim) - 1) {
        *state_it.offset(1, dim) = *mat_iter.state_it.offset(1, dim);
        *ls_it.offset(1, dim)    = *mat_iter.ls_it.offset(1, dim);
      }
    });

    *ls_it    = *mat_iter.ls_it;
    *state_it = *mat_iter.state_it;
    // TODO: Change this to synchronize();
    __syncthreads();

    // For each dimension, the value to choose is the one which has a smaller
    // levelset value (i.e the one towards the interface).
    auto extrap_data = Array<state_data_t, dims>{};
    for (auto i : range(band_width)) {
      const auto sign         = math::signum(*ls_it);
      const auto norm         = -sign * ls_it->norm(dh);
      const auto norm_mag_l1  = 0.0;

      const auto upper = (2.0 * i + 1.0) * dh * one_div_root_2;
      const auto lower = (2.0 * i - 1.0) * dh * one_div_root_2;

      if (*ls_it >= lower && *ls_it <= upper) {
        unrolled_for<dims>([&] (auto dim) {
          norm_mag_l1 += std::abs(norm[dim]);

          // Add the contibution for this dimension ...
          extrap_data[dim] = *ls_it.offset(-1, dim) < *ls_it.offset(1, dim)
            ? *state_it.offset(-1, dim) : *state_it.offset(1, dim);
        });
        
        *state_it = 0.0;
        unrolled_for<dims>([&] (auto dim) {
          *state_it += extrap_data * 
            (norm_mag_l1 > 1e-2 ? std::abs(norm[dim]) : 0.5);
        });
        if (norm_mag_l1 > 1e-2) {
          *state_it /= norm_mag_l1;
        }
      }
      // Make sure that the updated cell values are seen by the other threads
      // for the next iteration.
      __syncthreads();
    }

    *mat_iter.state_it = *state_it;
  }
};

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_GHOST_CELL_EXTRAPOLATOR_HPP
