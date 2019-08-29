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

#include <fluidity/algorithm/unrolled_for.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/traits/iterator_traits.hpp>
#include <fluidity/utility/constants.hpp>
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
  /// \param[in] mat_it        The type of the material iterator.
  /// \param[in] dh            The resolution of the material data grids.
  /// \tparam    MatIterator   The type of the material iterator.
  /// \tparam    T             The type of the resolution data.
  template <typename MatIterator, typename T>
  fluidity_host_device static void
  invoke(MatIterator&& mat_it, T dh, std::size_t band_width = 3) {
    static_assert(
      traits::is_material_iter_v<MatIterator>,
      "Extrapolation of ghost cells requires a material iterator!"
    );
    using value_t      = T;
    using mat_iter_t   = std::decay_t<MatIterator>;
    using state_iter_t = std::decay_t<decltype(mat_it.state_iterator())>;
    using ls_iter_t    = std::decay_t<decltype(mat_it.levelset_iterator())>;
    using state_data_t = std::decay_t<decltype(*mat_it.state_iterator())>;

    // The amount of padding for the loading. We need on element on each side
    // for each of the dimensions.
    constexpr auto pad  = std::size_t{1};
    constexpr auto dims = mat_iter_t::dimensions;

    // Shared memory iterators for the state and levelset data. Here we only
    // need a single padding element on each side.
    //auto ls_it    = 
    //  make_multidim_iterator<ls_iter_t, 1>(mat_it.levelset_iterator());
    //auto state_it = 
    //  make_multidim_iterator<state_iter_t, 1>(mat_it.state_iterator());

    // Offset the iterators ...
    unrolled_for<dims>([&] (auto dim) {
      //const auto offset = thread_id(dim) + 1;
      mat_it.shift(flattened_id(dim), dim);
      //ls_it.shift(offset, dim);
      //state_it.shift(offset, dim);
    });

    // Set the boundary data ...
//    unrolled_for<dims>([&] (auto dim) {
//      if (flattened_id(dim) == 0) {
//        *ls_it.offset(-1, dim) = std::numeric_limits<value_t>::max(); 
//      } else if (flattened_id(dim) >= mat_it.state_iterator().size(dim) - 1) {
//        *ls_it.offset(1, dim) = std::numeric_limits<value_t>::max(); 
//      } else if (thread_id(dim) == 0) {
//        //*state_it.offset(-1, dim) = *mat_it.state_iterator().offset(-1, dim);
//        *ls_it.offset(-1, dim)    = *mat_it.levelset_iterator().offset(-1, dim);
//      } else if (thread_id(dim) == block_size(dim) - 1) {
//        //*state_it.offset(1, dim) = *mat_it.state_iterator().offset(1, dim);
//        *ls_it.offset(1, dim)    = *mat_it.levelset_iterator().offset(1, dim);
//      }
//    });

//    *ls_it    = *mat_it.levelset_iterator();
    auto ls_it    = mat_it.levelset_iterator();
    auto state_it = mat_it.state_iterator();
    // TODO: Change this to synchronize();
    __syncthreads();

    // For each dimension, the value to choose is the one which has a smaller
    // levelset value (i.e the one towards the interface).
    auto extrap_data = Array<state_data_t, dims>{};
//    for (auto i : range(band_width)) {
    band_width += 3;
    int iters = std::abs(*ls_it) / dh + 1;
    for (auto j : range(1)) {
      // Need to handle the case that the normal is zero, when the interface is
      // exactly at the center of the cell ...
      constexpr auto tolerance = 1e-8;
      const auto sign        = math::signum(*ls_it);
      const auto norm        = -sign * ls_it.norm(dh);
      auto       norm_mag_l1 = 0.0;

      const auto upper = band_width * dh * cx::one_div_root_2;
      //const auto lower = dh * cx::one_div_root_2;
      const auto lower = dh;

      // NOTE: The levelset here needs to be positive so that we know we are
      //       outside of the material, hence in the region which needs to be
      //       extrapolated to.
      if (levelset::outside(ls_it) && *ls_it <= upper && *ls_it >= lower) {
        unrolled_for<dims>([&] (auto dim) {
          norm_mag_l1 += std::abs(norm[dim]);
        });

        using state_t    = std::decay_t<decltype(*state_it)>;
        bool keep_trying = true;
        auto p           = *state_it;
        auto q           = state_t{std::numeric_limits<value_t>::max()};
        using state_t    = std::decay_t<decltype(p)>;
        int i = 0;
        int dir = 0;
        while (keep_trying || iters-- >= 0) {
          i++;
          p = 0.0;
          // Compute the data to extrapolate, using the cell which is closer to
          // the interface ...
          unrolled_for<dims>([&] (auto dim) {
            // Add the contibution for this dimension ...
            dir = *ls_it.offset(-1, dim) < *ls_it.offset(1, dim) ? -1 : 1;
            extrap_data[dim] = *state_it.offset(dir, dim);
            p += extrap_data[dim] 
               * (norm_mag_l1 > 1e-2 ? std::abs(norm[dim]) : 0.5);
          });
        
          if (norm_mag_l1 > 1e-2) {
            p = p / norm_mag_l1;
          }

          //if (math::isnan(p)) {
          //  continue;
          //}

          auto err = std::abs(p[0] - q[0]);
          unrolled_for<state_t::elements - 1>([&] (auto e) {
            err += std::abs(p[e + 1] - q[e + 1]);
          });
          keep_trying = err >= tolerance || math::isnan(err);

          constexpr auto maxxx = 1e3;

          printf("A : %3lu : I : %3i, DIR: %3i : ER: %4.4f : NM : %4.4f : "
                 "K : %3lu "
                 "L : {%4.4f, %4.4f, %4.4f} "
                 "S : {%4.4f, %4.4f, %4.4f} "
                 "S1 : {%4.4f, %4.4f, %4.4f} "
                 "P : {%4.4f, %4.4f, %4.4f} "
                 "Q : {%4.4f, %4.4f, %4.4f}\n", 
            flattened_id(dim_x), i, dir, err, norm_mag_l1, keep_trying,
            *ls_it, *ls_it.offset(-1, 0), *ls_it.offset(1, 0),
            (*state_it)[0], (*state_it)[1], (*state_it)[2],
            (*state_it.offset(1, 0))[0], (*state_it.offset(1, 0))[1], 
            (*state_it.offset(1, 0))[2],
            p[0], p[1], p[2],
            min(q[0], maxxx), min(q[1], maxxx), min(q[2], maxxx)
          );
          // Set the old value to the new one.
          if (p != value_t{0} && !math::isnan(p)) {
            q = p;
            *mat_it.state_iterator() = p;
          } //else {
          //  q = *mat_it.state_iterator();
          //}
        }
        *mat_it.state_iterator() = p;
      }
      // Make sure that the updated cell values are seen by the other threads
      // for the next iteration.
    }

    // In the case that the data does not need to be extrapolated, this just
    // sets the state to its unmodified original value.
    //*mat_it.state_iterator() = *state_it;
  }
};

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_GHOST_CELL_EXTRAPOLATOR_HPP
