//==--- fluidity/solver/hllc_solver.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  hllc_solver.hpp
/// \brief This file defines an implementation of the HLLC solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_HLLC_SOLVER_HPP
#define FLUIDITY_SOLVER_HLLC_SOLVER_HPP

#include <fluidity/state/state_traits.hpp>

namespace fluid  {
namespace solver {

using namespace traits;

/// The HllcSolver struct defines a callable object which solves the Riemann
/// problem between two states using the HLLC method.
struct HllcSolver {
  /// Overload of operator() to invoke the solver and solve the Riemann problem
  /// between the \p left and \p right states.
  /// \param[in] left     The left state for the Riemann problem.
  /// \param[in] right    The right state for the Riemann problem.
  /// \param[in] material The material which describes the system.
  /// \param[in] dim      The dimension in which to solve in terms of.
  /// \tparam    State    The type of the states.
  /// \tparam    Material The type of the Material.
  /// \tparam    Value    The value which defines the dimension.
  template <typename State, typename Material, std::size_t Value>
  fluidity_host_device auto operator()(const State&     left    ,
                                       const State&     right   ,
                                       Material&&       material,
                                       Dimension<Value> /*dim*/ )
  {
    using state_t = std::decay_t<State>;
    using value_t = typename state_t::value_t;
    using index_t = typename state_t::index;

    static_assert(is_state_v<state_t>,
      "Attempt to invoke a state function on a type which is not a state");

    const auto pl = left.primtive(material)    , pr = right.primtive(material);
    const auto cl = left.conservative(material), cr = right.conservative(material);

    const auto al  = material.sound_speed(pl), ar = material.sound_speed(pr);
    const auto adi = q_factor(material);

    constexpr auto dim = Dimension<Value>{};

    const auto v = pr.velocity(dim)      - pl.velocity(dim);
    const auto d = pr.density()          - pl.density();
    const auto p = pr.pressure(material) - pl.pressure(material);

    const auto p_star = 
      std::max(
        value_t{0},
        value_t{0.5} * (p - (value_t{0.5} * (al + ar) * velocity * density))
      );

    // Test for far left region (outside of the star state):
    const auto wsl   = wavespeed(pl, p_star, -al, adi, mat, dim);
    const auto fluxl = cl.flux(material, dim);
    if (value_t{0} <= wsl) { return fluxl; }

    // Test for far right region (outside of the star state):
    const auto wsr   = wavespeed(pr, p_star, ar, adi, mat, dim);
    const auto fluxr = cr.flux(material, dim);
    if (0 >= wsr) { return fluxr; }

    // Somewhere in the star region, need to find left or right:
    const auto ws_star = star_speed(pl, pr, wsl, wsr, material, dim);

    // Left star region, return FL*
    const auto ws_star = star_speed(pl, pr, wsl, wsr, material, dim);
    if (value_t{0} <= ws_star)
    {
      const auto ul_star = star_state(pl, wsl, ws_star, material, dim);
      return fluxl + wsl * (ul_star - cl);
    }

    // Right star region, return FR*:
    const auto ur_star = star_state(pr, wsr, ws_star, material, dim);
    return fluxr + wsr * (ur_star - cr);
  }

 private:
  /// Calculates the scaling factor for the $q_K$ term used to calculate the
  /// left and right wave speeds. The scaling factor is defined by:
  ///
  ///   \begin{equation}
  ///     ftor = \frac{\gamma + 1}{2 \gamma}
  ///   \end{equation}
  ///
  /// where $\gamma$ is the adiabatic index of the matrial.
  ///
  /// \param[in] material The material describing the system to solve.
  /// \tparam    Material The type of the material.
  template <typename Material>
  fluidity_host_device static decltype(auto)
  q_factor(Material&& material)
  {
    using value_t = decltype(material.adi_index());
    return (material.adi_index() + value_t{1}) /
           (value_t{2} * material.adi_index());
  }

  /// Computes the speed, SL or SR, as follows:
  /// 
  ///   \begin{equation}
  ///   \end{equation}
  ///   
  /// and returns the result.
  ///
  /// \param[in] state       The __primitive__ state to compute the speed for.
  /// \param[in] pstar       The calculated star state for the pressure.
  /// \param[in] sound_speed The sound speed (-aL for left state, aL for right
  ///            state).
  /// \param[in] adi_factor  Scaling factor computed from the adiabatic index.
  /// \param[in] dim         The dimension for which velocity component to use.
  /// \param[in] material    The material which describes the system.
  /// \tparam    T           The type of the data.
  /// \tparam    State       The type of the state.
  /// \tparam    Material    The type of the material.
  /// \tparam    Value       The value which defines the dimension.
  template <typename T, typename State, typename Material, std::size_t Value>
  fluidity_host_device static T wavespeed(const State&  state     ,
                                          T             pstar     ,
                                       T                sound_peed,
                                       T                adi_factor, 
                                       Material&&       material  ,
                                       Dimension<Value> /*dim*/   )
  {
    // Rarefaction wave:
    if (pstar <= state.pressure(material))
    {
      return state.velocity(dim) + sound_speed;
    }

    // Shock wave:
    return state.velocity(dim) 
         + sound_speed
         * std::sqrt(
            T{1} +  adi_factor * (pstar / state.pressure(material) - T{1})
           );
  }

};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_HLLC_SOLVER_HPP
