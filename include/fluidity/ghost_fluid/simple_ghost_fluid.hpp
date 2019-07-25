//==--- fluidity/ghost_fluid/simple_ghost_fluid.hpp -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simple_ghost_fluid.hpp
/// \brief This file defines an implementation of the simplest ghost fluid
///        which simply copies the pressure and velocity from the other material
///        real cells into the ghost cells, and extrapolates the entropy from
///        the real fluid cells to the ghost fluid cells.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_SIMPLE_GHOST_FLUID_HPP
#define FLUIDITY_GHOST_FLUID_SIMPLE_GHOST_FLUID_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {

/// This struct implements the ghost fluid method, which takes a tuple of 
/// material iterator wrappers and loads ghost cell values into the state data
/// for a material using the levelsets for the materials.
///
/// This is the simplest method of setting the ghost cell data, and uses the
/// method of Fedkiw, et al (A Non-Oscillatory Eularian Approach to Interfaces
/// in Multi-Material Flows (The Ghost Fluid Method)) to set the ghost cells for
/// for material A by copying the pressure and velocity from the real cells in
/// material B and extrapolating the entropy from the real cells in material A
/// to the ghost cells for material A.
template <std::size_t Width>
struct SimpleGFM {
  /// Defines the width of the ghost fluid method (the number of cells which
  /// need to be set using the method).
  static constexpr auto width = Width;

  /// Function to invoke the GFM. This simply forwards the wrappers onto the
  /// appropriate implemetation based on the iterator types.
  /// \param[in] mat_it_wrappers The material iterator wrappers.
  /// \param[in] dh              The resolution of the material data grids.
  /// \tparam    MatItWrappers   The type of the iterator wrapper container.
  /// \tparam    T               The type of the resolution.
  template <typename MatItWrappers, typename T>
  fluidity_host_device static void
  invoke(MatItWrappers&& mat_it_wrappers, T dh) {
    constexpr auto dims = std::decay_t<
      decltype(get<0>(mat_it_wrappers).levelset_iterator())
    >::dimensions;
    
    // Start by offsetting the iterator wrappers to the current thread index ...
    for_each(mat_it_wrappers, [&] (auto& wrapper) {
      unrolled_for<dims>([&] (auto dim) {
        wrapper.shift(flattened_id(dim), dim);
      });
    });
    __syncthreads();

    auto print = [&] (auto message, auto v = 0.0f, int x = 0)
    {
      printf("{%03lu, %03lu} : {%03lu, %03lu} : {%03lu, %03lu} : %s, %012.10f, %3i\n",
        block_id(std::size_t{0})           , block_id(std::size_t{1}),
        thread_id(std::size_t{0})          , thread_id(std::size_t{1}),
        flattened_id(std::size_t{0}),
        flattened_id(std::size_t{1}), message, v, x);
    };

    // Next, if the cell is an interfacial cell, and inside the levelset, then
    // we need to set ghost cells for it, and we need to do this for each
    // material.
    int wrapper_index = 0;
    for_each(mat_it_wrappers, [&] (auto& wrapper) {
      wrapper_index++;
      const auto band = dh * static_cast<decltype(dh)>(width);
      auto ls_it      = wrapper.levelset_iterator();
      if (levelset::outside_interfacial_cell(ls_it, band) ||
          levelset::on_boundary(ls_it)) {
        unrolled_for<dims>([&] (auto dim) {
          // Find the material on the otherside of the boundary ...
          int other_wrapper_index = 0;
          for_each(mat_it_wrappers, [&] (auto& other_wrapper) {
            if (wrapper_index != ++other_wrapper_index) {
              auto other_ls_it = other_wrapper.levelset_iterator();
              if (levelset::inside_interfacial_cell(other_ls_it, band)) {
                // Determine if we need to walk to the right or to the left to
                // find the interface.
                const auto step = 
                  (*ls_it.offset(1, dim) < *ls_it)
                  ? int{1}
                  : int{-1};

                // Set the starting offset to try for the interface.
                auto off = step * static_cast<int>(
                  std::floor(std::abs(*ls_it / dh))
                );

                // Find the interface ...
                // TODO: There must be a better way to do this!
                auto entrop_ls_it = ls_it.offset(off, dim);
                const auto w      = dh * 1.25;
                while (!levelset::inside_interfacial_cell(entrop_ls_it, w)) {
                  entrop_ls_it.shift(step, dim);
                  off += step;
                }

                auto& mat_eos    = wrapper.eos();
                // The ghost state is the one that the data must be set for.
                // The real state is the one with real data to use to set the
                // ghost cells.
                auto ghost_state = wrapper.state_iterator()->primitive(mat_eos);
                auto real_state  = other_wrapper.state_iterator();

                // Copy pressure and velocity to the ghost cells:
                // Note that we don't need to convert the other state to
                // primitive because we are not setting it's values, just
                // extracting fluid properties directly, which is possible
                // from a state of either form.
                ghost_state.set_velocity(real_state->velocity(dim), dim);
                ghost_state.set_pressure(real_state->pressure(mat_eos));

                // Use the equation of state to get the density required for
                // the ghost state such that it has the same entropy as the
                // inside entropy state (i.e constant extrapolation of the
                // entropy from the inside interfacial cell to the ghost cell)
                const auto entrop_state = 
                  wrapper.state_iterator().offset(off, dim);
                const auto density      = mat_eos.density_for_const_entropy_log(
                    *entrop_state, ghost_state
                );
                ghost_state.set_density(density);

                // Convert the primitive ghost state back to conservative form.
                *wrapper.state_iterator() = ghost_state.conservative(mat_eos);
              }
            }
          });
        });
      }
    });
  }
};

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_SIMPLE_GHOST_FLUID_HPP

