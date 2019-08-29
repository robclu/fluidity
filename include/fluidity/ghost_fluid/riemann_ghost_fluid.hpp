//==--- fluidity/ghost_fluid/riemann_ghost_fluid.hpp ------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  riemann_ghost_fluid.hpp
/// \brief This file defines an implementation of the riemann ghost fluid. For
///        details of the specific implementation, refer to: ...
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_GHOST_FLUID_RIEMANN_GHOST_FLUID_HPP
#define FLUIDITY_GHOST_FLUID_RIEMANN_GHOST_FLUID_HPP

#include <fluidity/flux_method/flux_hllc.hpp>
#include <fluidity/math/interp.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace ghost {

/// This struct implements the Riemann Ghost Fluid Method (RGFM), which takes a
/// solves for the interfacial states given a material iterator.
///
/// This implemenation is the one presented in \cite{Sambasivan2009}, see the
/// paper for more info.
///
/// \tparam Width The width of the stencil required by the method.
template <std::size_t Width>
struct RiemannGFM {
  /// Defines the width of the ghost fluid method (the number of cells which
  /// need to be set using the method).
  static constexpr auto width = Width;

  /// Function to invoke the GFM. This simply forwards the wrappers onto the
  /// appropriate implemetation based on the iterator types.
  ///
  /// \pre The \p mat_iters have been offset to the correct cell, and don't need
  ///      to be offset in this method.
  ///
  /// \param[in] mat_iters    The material iterators.
  /// \param[in] dh           The resolution of the material data grids.
  /// \tparam    MatIterators The type of the iterator wrapper container.
  /// \tparam    T            The type of the resolution.
  template <typename MatIterators, typename T>
  fluidity_host_device static auto invoke(MatIterators&& mat_iters, T dh)
  -> void {
    constexpr auto three_over_two = T(3) / T(2);
    const auto     band           = dh;

    int  mat_id_1 = 0;
    for_each(mat_iters, [&] (auto& mat_iter) {
      using iter_t = std::decay_t<decltype(mat_iter)>;
      mat_id_1++;
      // Check if this cell an inside interfacial cell in the material, if it is
      // not, then we don't need to do anything.
      auto ls_it = mat_iter.levelset_iterator();

      if (levelset::inside_interfacial_cell(ls_it, dh) &&
          neighbour_not_on_boundary(ls_it)) {
        // Find the oher material ...
        int mat_id_2 = 0;
        for_each(mat_iters, [&] (auto& other_mat_iter) {
          mat_id_2++;
          auto other_ls_it = other_mat_iter.levelset_iterator();
          if (mat_id_2 != mat_id_1 && out_ifc_or_boundary(other_ls_it, dh)) {
            // STEP 1: Compute weights and riemann inputs ..
            const auto norm         = mat_iter.levelset_iterator().norm();
            const auto abs_phi      = std::abs(*mat_iter.levelset_iterator());
            const auto dh_three_two = T(1.5) * dh;

            // Make sure we are going into the material for the left state ..
            auto weights = norm * (abs_phi - dh_three_two) / dh;
            auto w_l     = math::interp(
              mat_iter.state_iterator(), weights
            );

            printf(
              "A {%3lu} : {%3i, %3i} : N : {%4.6f} : APHI : {%4.6f} "
              "WE : %4.6f "
              "WL : %4.6f, %4.6f, %4.6f\n",
              flattened_id(0),
              mat_id_1,
              mat_id_2,
              norm[0],
              abs_phi,
              weights[0],
              w_l.density(), w_l.pressure(mat_iter.eos()), w_l.velocity(dim_x)
            );

            // Make sure we are going out of the material for the right state ..
            weights  = norm * (abs_phi + dh_three_two) / dh;
            auto w_r = math::interp(
              other_mat_iter.state_iterator(), weights
            );

            printf(
              "B {%3lu} : {%3i, %3i} : N : {%4.6f} : APHI : {%4.6f} "
              "WE : %4.6f "
              "WR : %4.6f, %4.6f, %4.6f\n",
              flattened_id(0),
              mat_id_1,
              mat_id_2,
              norm[0],
              abs_phi,
              weights[0],
              w_r.density(), w_r.pressure(other_mat_iter.eos()), w_r.velocity(dim_x)
            );

            // STEP 2: Rotate the velocities into the normal and tangential
            //         directions ...
            // TODO: Change this to use rotation matrix ...
            const auto vl       = w_l.velocity_vec();
            const auto vr       = w_r.velocity_vec();
            const auto vl_dot_n = math::dot(w_l.velocity_vec(), norm);
            const auto vr_dot_n = math::dot(w_r.velocity_vec(), norm);
            w_l.set_velocity(vl_dot_n, dim_x);
            w_r.set_velocity(vr_dot_n, dim_x);

            // Reset the tangential velocities ...
            unrolled_for<iter_t::dimensions - 1>([&] (auto dim) {
              w_l.set_velocity(0, dim + 1);
              w_r.set_velocity(0, dim + 1);
            });

            // STEP 3: Solve the riemann problem to find the star state. The
            //         star state that we need is the left star state because we
            //         have defined left to be inside the material.
            auto star_state = flux::Hllc::solve_star_conservative_left(
              w_l, w_r, mat_iter.eos(), other_mat_iter.eos(), dim_x
            ); 

            // STEP 4: Rotate the star state velocities back into the (u,v,w)
            //         coordinate system.
            // This is the same as:
            //  ul* = u^{n*}\hat{n} + \bld{u}_L^{t} where
            //  \bld{u}_L^{t} = \bld{u}_L - (\bld{u}_L \dot \hat{n})\hat{n}
            unrolled_for<iter_t::dimensions>([&] (auto dim) {
              star_state.set_velocity(
                vl[dim] + (star_state.velocity(dim) - vl_dot_n) * norm[dim],
                dim
              );
            });

            
            auto ssl = flux::Hllc::solve_star_conservative_left(
              w_l, w_r, mat_iter.eos(), other_mat_iter.eos(), dim_x
            );
            
            auto ssr = flux::Hllc::solve_star_conservative_right(
              w_l, w_r, mat_iter.eos(), other_mat_iter.eos(), dim_x
            );

            printf("C {%3lu} : {%3i, %3i} "
                "WL : %4.6f, %4.6f, %4.6f "
                "WR : %4.6f, %4.6f, %4.6f "
                "MS : %4.6f, %4.6f, %4.6f "
                "SSL : %4.6f, %4.6f, %4.6f "
                "SSR : %4.6f, %4.6f, %4.6f\n",
              flattened_id(0),
              mat_id_1,
              mat_id_2,
              w_l.density(), w_l.pressure(mat_iter.eos()), w_l.velocity(dim_x),
              w_r.density(), w_r.pressure(other_mat_iter.eos()), w_r.velocity(dim_x),
              mat_iter.state_iterator()->density(),
              mat_iter.state_iterator()->pressure(mat_iter.eos()),
              mat_iter.state_iterator()->velocity(dim_x),
              ssl.density(),
              ssl.pressure(mat_iter.eos()),
              ssl.velocity(dim_x),
              ssr.density(),
              ssr.pressure(mat_iter.eos()),
              ssr.velocity(dim_x)
            );

            // Finally, set the cell to have the star state value,
            // but we need to set the cell next to the inside interfacial cell,
            // not the inside interfacial cell, because we want to preserve that
            // data.
            auto state_it = mat_iter.state_iterator();
            unrolled_for<iter_t::dimensions>([&] (auto dim) {
              state_it.shift(math::signum(norm[dim]), dim);
            });
            *state_it = star_state;
            //*mat_iter.state_iterator() = star_state;
          }
        });
      }
    });
  }


 private:
  /// Checks if the neighbouring cells are not on the boundary from the \p
  /// it. Returns true if non of the neighbouring cells are on the boundary, in 
  /// any direction, otherwise returns false.
  /// \param[in] it       The iterator to check.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_host_device static auto neighbour_not_on_boundary(Iterator&& it)
  -> bool {
    using iter_t = std::decay_t<Iterator>;
    for (auto dim : range(iter_t::dimensions)) {
      if (levelset::on_boundary(it.offset(-1, dim)) ||
          levelset::on_boundary(it.offset(1 , dim))) {
        return false;
      }
    }
    return true;
  }

  /// Returns true if the \p ls_it levelset iterator is either an outside
  /// interfacial cell or is on the boundary, otherwise returns false.
  /// \param[in] ls_it            The levelset iterator to check.
  /// \param[in] width            The width of the interfacial band.
  /// \tparam    LevelsetIterator The type of the levelset iterator.
  /// \tparam    T                The type of the bandwidth. 
  template <typename LevelsetIterator, typename T>
  fluidity_host_device static auto out_ifc_or_boundary(
    LevelsetIterator&& ls_it, T width 
  ) -> bool {
    return levelset::outside_interfacial_cell(ls_it, width)
     || levelset::on_boundary(ls_it);
  }
};

}} // namespace fluid::ghost

#endif // FLUIDITY_GHOST_FLUID_RIEMANN_GHOST_FLUID_HPP

