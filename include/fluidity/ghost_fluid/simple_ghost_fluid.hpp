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
struct SimpleGFM {
  /// Function to invoke the GFM. This simply forwards the wrappers onto the
  /// appropriate implemetation based on the iterator types.
  /// \param[in] mat_it_wrappers The material iterator wrappers.
  /// \tparam    MatItWrappers   The type of the iterator wrapper container.
  template <typename MatItWrappers>
  fluidity_device_host static void invoke(MatItWrappers&& mat_it_wrappers)
  {
    constexpr auto dims = 
      std::decay_t<decltype(get<0>(mat_it_wrappers).ls_it())::num_dimensions;
    
    // Start by offsetting the iterator wrappers to the current thread index ...
    unrolled_for<dims>([&] (auto dim)
    {
      for_each(mat_it_wrappers, [&] fluidity_device_host (auto& wrapper)
      {
        wrapper.ls_it.shift(dim, flattened_id(dim));
        wrapper.state_it.shift(dim, flattened_id(dim));
      });
    });
    __syncthreads();

    bool mat_set = false;
    // Next, if the cell is an interfacial cell, and inside the levelset, then
    // we need to set ghost cells for it, and we need to do this for each
    // material.
    for_each(mat_it_wrappers, [&] (auto& wrapper)
    {
      if (levelset::inside_boundary_cell(*(wrapper.ls_it), dh) && !mat_set)
      {
        mat_set = true;
        auto& mat_eos = wrapper.eos;
        unrolled_for<dims>([&] (auto dim)
        {
          auto mat_used = false;
          // Find the material on the otherside of the boundary ...
          for_each(mat_it_wrappers, [&] (auto& other_wrapper)
          {
            for (auto off : {{ -1, 1 }})
            { 
              auto other = *(other_wrapper.ls_it.offset(dim, off))
              if (levelset::outside_boundary_cell(other, dh) && !mat_used)
              {
                mat_used = true;
                // Set the ghost fluid state data.
                for (auto w : range(off, off * width, off))
                {
                  // Copy pressure and velocity
                  auto entrop_state = wrapper.state_it;
                  auto set_state    = entrop_state.offset(dim, w);
                  auto setter_state = other_wrapper.state_it.offset(dim, w);

                  set_state->set_pressure(setter_state->pressure(), mat_eos);
                  set_state->set_velocity(dim, setter_state->pressure());
                  set_state->set_entropy(*entrop_state, mat_eos);
                }
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

