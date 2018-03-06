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
#include <cmath>

namespace fluid  {
namespace solver {

using namespace ::fluid::state::traits;

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
        value_t{0.5} * (p - (value_t{0.5} * (al + ar) * v * d))
      );

    // Test for far left region (outside of the star state):
    const auto wsl   = wavespeed(pl, p_star, -al, adi, material, dim);
    const auto fluxl = cl.flux(material, dim);
    if (value_t{0} <= wsl) { return fluxl; }

    // Test for far right region (outside of the star state):
    const auto wsr   = wavespeed(pr, p_star, ar, adi, material, dim);
    const auto fluxr = cr.flux(material, dim);
    if (0 >= wsr) { return fluxr; }

    // Somewhere in the star region, need to find left or right:
    const auto ws_star = star_speed(pl, pr, wsl, wsr, material, dim);

    // Left star region, return FL*
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
  fluidity_host_device static T wavespeed(const State&     state     ,
                                          T                pstar     ,
                                          T                sound_peed,
                                          T                adi_factor, 
                                          Material&&       material  ,
                                          Dimension<Value> /*dim*/   )
  {
    using state_t = std::decay_t<State>;
    static_assert(state_t::format == state::FormType::primitive,
                  "Wavespeed computation requires primitive state vector.");

    // Define the dimension to ensure constexpr functionality:
    constexpr auto dim = Dimension<Value>{};

    const auto sound_speed = material.sound_speed(state);
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

  /// Computes the wavespeed in the star region, S*, as follows:
  ///
  ///   \begin{equation}
  ///   \end{equation}
  ///   
  /// and returns the result.
  ///
  /// \param[in] statel     The __primitve__ left state vector.
  /// \param[in] stater     The __primitive__ right state vector.
  /// \param[in] wavespeedl The left wave speed: SL.
  /// \param[in] wavespeedr The right wave speed: SR.
  /// \param[in] mat        The material describing the system.
  /// \param[in] dim        The dimension to use the velocity component of.
  /// \tparam    T          The type of the data.
  /// \tparam    State      The type of the states.
  /// \tparam    Material   The type of the material.
  /// \tparam    Value      The value which defines the dimension.
  template <typename T, typename State, typename Material, std::size_t Value>
  fluidity_host_device static T star_speed(const State&     statel    ,
                                           const State&     stater    ,
                                           T                wavespeedl,
                                           T                wavespeedr,
                                           Material&&       material  ,
                                           Dimension<Value> /*dim*/   )
  {
    // Define the dimension to ensure constexpr functionality.
    constexpr auto dim = Dimension<Value>{};

    // Compute the factor: $\rho * (S-k - u_k)
    const auto factorl = statel.density() * (wavespeedl - statel.velocity(dim));
    const auto factorr = stater.density() * (wavespeedr - stater.velocity(dim));

    // The computation is the following (dX = Density x):
    //    pR - pL + dL * vL(SL - vL) - dR * vR(SR - vR)
    //    ---------------------------------------------
    //              dL(SL - vL) - dR(SR - vR)
    return ((stater.pressure(material) - statel.pressure(material))            +
            (statel.velocity(dim) * factorl) - (stater.velocity(dim) * factorr))
           / (factorl - factorr);  
  }

  /// Computes the star state, UL* or UR*, given by:
  /// 
  ///   /begin{equation}
  ///   /end{equation}
  /// 
  /// and returns the result.
  ///
  /// \param[in] state       The __primitive__ state to compute the conservative
  ///            star state from.
  /// \param[in] state_speed The max wave speed Sk for the state in this 
  ///            direction, k.
  /// \param[in] star_speed  The star speed.
  /// \param[in] dim         The dimension to use the velocity component of.
  /// \tparam    T           The data type being used.
  /// \tparam    State       The type of the state.
  /// \tparam    Material    The type of material.
  /// \tparam    Value       The value which defines the dimension.
  template <typename T, typename State, typename Material, std::size_t Value>
  fluidity_host_device static auto star_state(const State&     state      ,
                                              T                state_speed,
                                              T                star_speed ,
                                              Material&&       material   ,
                                              Dimension<Value> /*dim*/    )
  {
    using state_t  = std::decay_t<State>;
    using index_t  = typename state_t::index;
    using vector_t = Array<typename state_t::value_t, state_t::elements>;

    // Define the dimension to ensure constexpr functionality.
    constexpr auto dim = Dimension<Value>{};

    const auto state_factor = state.density()
                            * (state_speed - state.velocity(dim));
    const auto scale_factor = state_factor / (state_speed - star_speed);
    const auto e_div_rho    = T{0.5} * state.v_squared_sum()
                            + material.eos(state);

    // Create a vector which is initially the state scaled by the scaling factor
    // , which is the default value for most of the components, and then modify
    // the appropriate other elements.
    vector_t temp(scale_factor * state);
    temp[index_t::density]       = scale_factor;
    temp[index_t::velocity(dim)] = scale_factor * star_speed;
    temp[index_t::pressure]      =
        scale_factor
      * (e_div_rho
      +  (star_speed - state.velocity(dim))
      *  (star_speed + state.pressure(material) / scale_factor)
        );
    return temp;
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_HLLC_SOLVER_HPP
