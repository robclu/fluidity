//==--- fluidity/flux_method/flux_hllc.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_hllc.hpp
/// \brief This file defines an implementation of the computation of the flux
///        between two states solve the Riemann problem between them using the
///        HLLC method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_FLUX_METHOD_FLUX_HLLC_HPP
#define FLUIDITY_FLUX_METHOD_FLUX_HLLC_HPP

#include <fluidity/container/array.hpp>
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid {
namespace flux  {

/// The Hllc struct defines a callable object which computes the flux between
/// two conservative form states by solving the Riemann problem between them.
/// The implementation is only defined for conservative states so that no
/// conversion is performed in the solver, since it can be performed before
/// invoking the method. For more information, see Toro, 
struct Hllc {
 private:
  /// The Impl struct provides the implementation of the LF flux method for a
  /// given material type in a specific dimension.
  /// \tparam Mat The type of the material used for the flux computation.
  /// \tparam Dim The type of the dimension.
  template <typename Mat, typename Dim>
  struct Impl {
   private:
    /// Defines the type of material used for the flux computation.
    using material_t = std::decay_t<Mat>;
    /// Defines the data type used in the implementation.
    using value_t    = typename material_t::value_t;

    /// Defines the dimension which the implementation solves for.
    static constexpr auto dim = Dim();

    /// Stores a reference to the material to use for the implementation.
    material_t _mat;   //!< The material to solve for.
    value_t    _dtdh;  //!< Time delta for the iteration.

    /// Calculates the scaling factor for the $q_K$ term used to calculate the
    /// left and right wave speeds. The scaling factor is defined by:
    ///
    ///   \begin{equation}
    ///     ftor = \frac{\gamma + 1}{2 \gamma}
    ///   \end{equation}
    ///
    /// where $\gamma$ is the adiabatic index of the matrial.
    fluidity_host_device auto q_factor() const -> value_t {
      return (_mat.adiabatic() + value_t{1}) /
             (value_t{2} * _mat.adiabatic());
    }

    /// Computes the speed, SL or SR, as follows:
    /// 
    ///   \begin{equation}
    ///   \end{equation}
    ///   
    /// and returns the result.
    ///
    /// \param[in] state       The __conservative__ state.
    /// \param[in] pstar       The calculated star state for the pressure.
    /// \param[in] sound_speed Sound speed (-aL:left state, aR:right state).
    /// \param[in] adi_scale   Scaling factor computed from the adiabatic index.
    /// \tparam    State       The type of the state.
    template <typename State>
    fluidity_host_device auto wavespeed(
      const State& state      ,
      value_t      pstar      ,
      value_t      sound_speed,
      value_t      adi_scale 
    ) const -> value_t {
      const auto pressure = state.pressure(_mat);
      // Rarefaction wave:
      if (pstar <= pressure) {
        return state.velocity(dim) + sound_speed;
      } 
    
      // Shock wave:
      constexpr auto one = value_t{1};
      return state.velocity(dim) 
        + sound_speed 
        * std::sqrt(one + adi_scale * (pstar / pressure - one));
    }

    /// Computes the wavespeed in the star region, S*, as follows:
    ///
    ///   \begin{equation}
    ///   \end{equation}
    ///   
    /// and returns the result.
    ///
    /// \param[in] statel     The __conservative__ left state vector.
    /// \param[in] stater     The __conservative__ right state vector.
    /// \param[in] wavespeedl The left wave speed: SL.
    /// \param[in] wavespeedr The right wave speed: SR.
    /// \tparam    State      The type of the states.
    template <typename State>
    fluidity_host_device auto star_speed(
      const State& statel    ,
      const State& stater    ,
      value_t      wavespeedl,
      value_t      wavespeedr
    ) const -> value_t {
      const auto vl = statel.velocity(dim);
      const auto vr = stater.velocity(dim);

      // Compute the factor: $\rho * (S-k - u_k)
      const auto factorl = statel.density() * (wavespeedl - vl);
      const auto factorr = stater.density() * (wavespeedr - vr);

      // The computation is the following (dX = Density x):
      //    pR - pL + dL * vL(SL - vL) - dR * vR(SR - vR)
      //    ---------------------------------------------
      //              dL(SL - vL) - dR(SR - vR)
      return (
        stater.pressure(_mat) -
        statel.pressure(_mat) +
        vl * factorl          - 
        vr * factorr          
      ) / (factorl - factorr);
    } 

    /// Computes the star state, UL* or UR*, given by:
    /// 
    ///   /begin{equation}
    ///   /end{equation}
    /// 
    /// and returns the result.
    ///
    /// \param[in] state       The __conservative__ input state.
    /// \param[in] state_speed The max wave speed, Sk, in direction k.
    /// \param[in] star_speed  The wave speed in the star region.
    /// \tparam    State       The type of the state.
  template <typename State>
  fluidity_host_device auto star_state(
    const State& state      ,
    value_t      state_speed,
    value_t      star_speed 
  ) const -> Array<typename State::value_t, State::dimensions> {
    using state_t   = std::decay_t<State>;
    using value_t   = typename state_t::value_t;
    using indexer_t = typename state_t::index;
    using vector_t  = Array<value_t, state_t::elements>;

    // Factors used to set the flux values. Stored here so that they only have
    // to be computed once.
    const auto state_factor = state_speed - state.velocity(dim);
    const auto scale_factor = state_factor / (state_speed - star_speed);

    // Create a vector which is initially the state scaled by the scaling factor
    // because that's the default value for most of the components:
    vector_t u = scale_factor * state;

    // Set the velocity component:
    //  = rho_k * ((S_k - u_k) / (S_k - S_*)) * S_*
    u[indexer_t::velocity(dim)] = star_speed * state.density() * scale_factor;

    // Set the energy component:
    //  = rho_k * ((S_k - u_k) / (S_k - S_*)) * S_* 
    //      [ E_k / rho_k + (S_* - u_k) * [S_* + (p_k / (rho_k *(S_k - u_k)))]
    //
    // Currently stored is \rho_k * U_k, so add the rest:
    u[indexer_t::energy] =
      scale_factor                        *
      (state.energy(_mat)                 +
       state.density()                    *
       (star_speed - state.velocity(dim)) *
       (star_speed + state.pressure(_mat) / (state.density() * state_factor))
      );

    return u;
  }

 public:
    /// Constructor which sets the material and scaling factor.
    /// \param[in] mat  The material for the computation.
    /// \param[in] dtdh The scaling factor for the computation.
    fluidity_host_device Impl(material_t mat, value_t dtdh)
    : _mat(mat), _dtdh(dtdh) {}

    /// Overload of operator() to compute the flux between the \p left and 
    /// \p right conservative states.
    /// \param[in] ul     The left state for the Riemann problem.
    /// \param[in] ur     The right state for the Riemann problem.
    /// \tparam    State  The type of the states.
    template <typename State>
    fluidity_host_device auto operator()(
      const State& ul,
      const State& ur
    ) const {
      const auto al  = _mat.sound_speed(ul);
      const auto ar  = _mat.sound_speed(ur);
      const auto adi = q_factor();

      const auto p_pvrs =
        value_t{0.5} * (
          (ul.pressure(_mat) + ur.pressure(_mat))
          - (
            value_t{0.25} 
              * (ur.velocity(dim) - ul.velocity(dim))
              * (ur.density()     + ul.density())
              * (al + ar)
          )
        );
    
      // Pressure in the star region, as per Toro, Equation 10.67, page 331:
      const auto p_star = std::max(value_t{0}, p_pvrs);

      // Test for far left region (outside of the star state):
      const auto wsl   = wavespeed(ul, p_star, -al, adi);
      const auto fluxl = ul.flux(_mat, dim);
      if (value_t{0} <= wsl) { return fluxl; }

      // Test for far right region (outside of the star state):
      const auto wsr   = wavespeed(ur, p_star, ar, adi);
      const auto fluxr = ur.flux(_mat, dim);
      if (0 >= wsr) { return fluxr; }

      // Somewhere in the star region, need to find left or right:
      const auto ws_star = star_speed(ul, ur, wsl, wsr);

      // Left star region, return FL*
      if (value_t{0} <= ws_star) {
        return fluxl + wsl * (star_state(ul, wsl, ws_star) - ul);
      }

      // Right star region, return FR*:
      return fluxr + wsr * (star_state(ur, wsr, ws_star) - ur);
    }
  };

  /// Gets an instance of the flux evaluator for a give material \p mat and a
  /// given dimension.
  /// \param[in] mat      The material to use for the flux method.
  /// \param[in] dtdh     The space and time discretization factor.
  /// \tparam    T        The type of the scaling factor.
  /// \tparam    Mat      The type of the material.
  /// \tparam    Dim      The type of the dimension.
  template <typename Mat, typename T, typename Dim>
  fluidity_host_device static auto get(const Mat& mat, T dtdh, Dim) {
    using flux_impl_t = Impl<Mat, Dim>;
    return flux_impl_t{mat, dtdh};
  }


 public:
  /// Overload of operator() to compute the flux between the \p left and 
  /// \p right conservative states, where the states might be from different
  /// materials.
  /// \param[in] ul       The left state for the Riemann problem.
  /// \param[in] ur       The right state for the Riemann problem.
  /// \param[in] matl     The material for the left state.
  /// \param[in] matr     The material for the right state.
  /// \tparam    State    The type of the states.
  /// \tparam    MatLeft  The type of the material for the left state.
  /// \tparam    MatRight The type of the material for the right state.
  template <typename State, typename EosLeft, typename EosRight, typename Dim>
  fluidity_host_device static auto solve_star_conservative(
    const State& ul  ,
    const State& ur  ,
    EosLeft&&    eosl,
    EosRight&&   eosr,
    Dim          dim = dimx_t()
  ) -> State {
    using value_t = typename std::decay_t<EosLeft>::value_t;
    const auto al = eosl.sound_speed(ul);
    const auto ar = eosr.sound_speed(ur);

    const auto p_pvrs =
      value_t{0.5} * (
        (ul.pressure(eosl) + ur.pressure(eosr))
        - (
          value_t{0.25} 
            * (ur.velocity(dim) - ul.velocity(dim))
            * (ur.density()     + ul.density())
            * (al + ar)
        )
      );
    
    // Pressure in the star region, as per Toro, Equation 10.67, page 331:
    const auto p_star = std::max(value_t{0}, p_pvrs);

    // Test for far left region (outside of the star state):
    const auto wsl = wavespeed_slr(ul, p_star, -al, q_k(eosl), eosl, dim);
    if (value_t{0} <= wsl) { 
      return ul;
    }

    // Test for far right region (outside of the star state):
    const auto wsr = wavespeed_slr(ur, p_star, ar, q_k(eosr), eosr, dim);
    if (0 >= wsr) {
      return ur;
    }

    // Somewhere in the star region, need to find left or right:
    const auto ws_star = wavespeed_star(ul, ur, wsl, wsr, eosl, eosr, dim);

    // Return either the left or right star state based on the star speed...
    return value_t{0} <= ws_star
      ? star_state_cons(ul, wsl, ws_star, eosl, dim)
      : star_state_cons(ur, wsr, ws_star, eosr, dim);
  }

  /// Overload of operator() to compute the flux between the \p left and 
  /// \p right conservative states, where the states might be from different
  /// materials.
  /// \param[in] ul       The left state for the Riemann problem.
  /// \param[in] ur       The right state for the Riemann problem.
  /// \param[in] matl     The material for the left state.
  /// \param[in] matr     The material for the right state.
  /// \tparam    State    The type of the states.
  /// \tparam    MatLeft  The type of the material for the left state.
  /// \tparam    MatRight The type of the material for the right state.
  template <typename State, typename EosLeft, typename EosRight, typename Dim>
  fluidity_host_device static auto solve_star_conservative_left(
    const State& ul  ,
    const State& ur  ,
    EosLeft&&    eosl,
    EosRight&&   eosr,
    Dim          dim = dimx_t()
  ) -> State {
    using value_t = typename std::decay_t<EosLeft>::value_t;
    const auto al = eosl.sound_speed(ul);
    const auto ar = eosr.sound_speed(ur);

    const auto p_pvrs =
      value_t{0.5} * (
        (ul.pressure(eosl) + ur.pressure(eosr))
        - (
          value_t{0.25} 
            * (ur.velocity(dim) - ul.velocity(dim))
            * (ur.density()     + ul.density())
            * (al + ar)
        )
      );
    
    // Pressure in the star region, as per Toro, Equation 10.67, page 331:
    const auto p_star = std::max(value_t{0}, p_pvrs);

    // Test for far left region (outside of the star state):
    const auto wsl = wavespeed_slr(ul, p_star, -al, q_k(eosl), eosl, dim);
    const auto wsr = wavespeed_slr(ur, p_star, ar, q_k(eosr), eosr, dim);
    // Somewhere in the star region, need to find left or right:
    const auto ws_star = wavespeed_star(ul, ur, wsl, wsr, eosl, eosr, dim);

    // Return either the left or right star state based on the star speed...
    return star_state_cons(ul, wsl, ws_star, eosl, dim);
  }

  /// Overload of operator() to compute the flux between the \p left and 
  /// \p right conservative states, where the states might be from different
  /// materials.
  /// \param[in] ul       The left state for the Riemann problem.
  /// \param[in] ur       The right state for the Riemann problem.
  /// \param[in] matl     The material for the left state.
  /// \param[in] matr     The material for the right state.
  /// \tparam    State    The type of the states.
  /// \tparam    MatLeft  The type of the material for the left state.
  /// \tparam    MatRight The type of the material for the right state.
  template <typename State, typename EosLeft, typename EosRight, typename Dim>
  fluidity_host_device static auto solve_star_conservative_right(
    const State& ul  ,
    const State& ur  ,
    EosLeft&&    eosl,
    EosRight&&   eosr,
    Dim          dim = dimx_t()
  ) -> State {
    using value_t = typename std::decay_t<EosLeft>::value_t;
    const auto al = eosl.sound_speed(ul);
    const auto ar = eosr.sound_speed(ur);

    const auto p_pvrs =
      value_t{0.5} * (
        (ul.pressure(eosl) + ur.pressure(eosr))
        - (
          value_t{0.25} 
            * (ur.velocity(dim) - ul.velocity(dim))
            * (ur.density()     + ul.density())
            * (al + ar)
        )
      );
    
    // Pressure in the star region, as per Toro, Equation 10.67, page 331:
    const auto p_star = std::max(value_t{0}, p_pvrs);

    // Test for far right region (outside of the star state):
    const auto wsl = wavespeed_slr(ul, p_star, -al, q_k(eosl), eosl, dim);
    const auto wsr = wavespeed_slr(ur, p_star, ar, q_k(eosr), eosr, dim);

    // Somewhere in the star region, need to find left or right:
    const auto ws_star = wavespeed_star(ul, ur, wsl, wsr, eosl, eosr, dim);

    // Return either the left or right star state based on the star speed...
    return star_state_cons(ur, wsr, ws_star, eosr, dim);
  }

 private:
  //==--- [Material agnostic] ----------------------------------------------==//

  /// Calculates the scaling factor for the $q_K$ term used to calculate the
  /// left and right wave speeds. The scaling factor is defined by:
  ///
  ///   \begin{equation}
  ///     ftor = \frac{\gamma + 1}{2 \gamma}
  ///   \end{equation}
  ///
  /// where $\gamma$ is the adiabatic index of the matrial.
  /// \param[in] mat The material to use to compute the factor.
  template <typename Eos>
  fluidity_host_device static auto q_k(Eos&& eos) 
  -> typename std::decay_t<Eos>::value_t {
    using value_t = typename std::decay_t<Eos>::value_t;
    return (eos.adiabatic() + value_t{1})
         / (value_t{2} * eos.adiabatic());
  }

  /// Computes the speed, SL or SR, as follows:
  /// 
  ///   \begin{equation}
  ///   \end{equation}
  ///   
  /// and returns the result.
  ///
  /// \param[in] state       The __conservative__ state.
  /// \param[in] pstar       The calculated star state for the pressure.
  /// \param[in] sound_speed Sound speed (-aL:left state, aR:right state).
  /// \param[in] adi_scale   Scaling factor computed from the adiabatic index.
  /// \param[in] mat         The material to ues for the computation.
  /// \tparam    State       The type of the state.
  /// \tparam    Mat         The type of the material.
  template <typename State, typename T, typename Eos, typename Dim>
  fluidity_host_device static auto wavespeed_slr(
    const State& state      ,
    T            pstar      ,
    T            sound_speed,
    T            adi_scale  ,
    Eos&&        eos        ,
    Dim          dim
  ) -> T {
    const auto pressure = state.pressure(eos);
    // Rarefaction wave:
    if (pstar <= pressure) {
      return state.velocity(dim) + sound_speed;
    } 

    // Shock wave:
    return state.velocity(dim) 
      + sound_speed
      * std::sqrt(T(1) + adi_scale * (pstar / pressure - T(1)));
  }

  /// Computes the wavespeed in the star region, S*, as follows:
  ///
  ///   \begin{equation}
  ///   \end{equation}
  ///   
  /// and returns the result.
  ///
  /// \param[in] statel     The __conservative__ left state vector.
  /// \param[in] stater     The __conservative__ right state vector.
  /// \param[in] wavespeedl The left wave speed: SL.
  /// \param[in] wavespeedr The right wave speed: SR.
  /// \param[in] matl       The left material to ues for the computation.
  /// \param[in] matr       The right material to ues for the computation.
  /// \tparam    State      The type of the states.
  /// \tparam    MatLeft    The type of the left material.
  /// \tparam    MatRight   The type of the right material.
  template <
    typename State   ,
    typename T       ,
    typename EosLeft ,
    typename EosRight,
    typename Dim
  >
  fluidity_host_device static auto wavespeed_star(
    const State& statel    ,
    const State& stater    ,
    T            wavespeedl,
    T            wavespeedr,
    EosLeft&&    eosl      ,
    EosRight&&   eosr      ,
    Dim          dim 
  ) -> T {
    const auto vl = statel.velocity(dim);
    const auto vr = stater.velocity(dim);

    // Compute the factor: $\rho * (S-k - u_k)
    const auto factorl = statel.density() * (wavespeedl - vl);
    const auto factorr = stater.density() * (wavespeedr - vr);

    // The computation is the following (dX = Density x):
    //    pR - pL + dL * vL(SL - vL) - dR * vR(SR - vR)
    //    ---------------------------------------------
    //              dL(SL - vL) - dR(SR - vR)
    return (
      stater.pressure(eosr) -
      statel.pressure(eosl) +
      vl * factorl          - 
      vr * factorr          
    ) / (factorl - factorr);
  } 

  /// Computes the star state, UL* or UR*, given by:
  /// 
  ///   /begin{equation}
  ///   /end{equation}
  /// 
  /// and returns the result.
  ///
  /// \param[in] state       The __conservative__ input state.
  /// \param[in] state_speed The max wave speed, Sk, in direction k.
  /// \param[in] star_speed  The wave speed in the star region.
  /// \param[in] mat         The material for the computation.
  /// \tparam    State       The type of the state.
  /// \tparam    Mat         The type of the material for the computation.
  template <typename State, typename T, typename Eos, typename Dim>
  fluidity_host_device static auto star_state_cons(
    const State& state      ,
    T            state_speed,
    T            star_speed ,
    Eos&&        eos        ,
    Dim          dim
  ) -> State {
    using state_t   = std::decay_t<State>;
    using value_t   = typename state_t::value_t;
    using indexer_t = typename state_t::index;
    using vector_t  = Array<value_t, state_t::elements>;

    // Factors used to set the flux values. Stored here so that they only have
    // to be computed once.
    const auto state_factor = state_speed - state.velocity(dim);
    const auto scale_factor = state_factor / (state_speed - star_speed);

    // Create a vector which is initially the state scaled by the scaling factor
    // because that's the default value for most of the components:
    vector_t u = scale_factor * state;

    // Set the velocity component:
    //  = rho_k * ((S_k - u_k) / (S_k - S_*)) * S_*
    u[indexer_t::velocity(dim)] = star_speed * state.density() * scale_factor;

    // Set the energy component:
    //  = rho_k * ((S_k - u_k) / (S_k - S_*)) * S_* 
    //      [ E_k / rho_k + (S_* - u_k) * [S_* + (p_k / (rho_k *(S_k - u_k)))]
    //
    // Currently stored is \rho_k * U_k, so add the rest:
    u[indexer_t::energy] =
      scale_factor                        *
      (state.energy(eos)                  +
       state.density()                    *
       (star_speed - state.velocity(dim)) *
       (star_speed + state.pressure(eos) / (state.density() * state_factor))
      );

    return State{u};
  }
};

}} // namespace fluid::flux

#endif // F
