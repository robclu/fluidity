//==--- fluidity/flux_method/flux_force.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_force.hpp
/// \brief This file defines an implementation of the computation of the flux
///        between two states using the FORCE
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_FLUX_METHOD_FLUX_FORCE_HPP
#define FLUIDITY_FLUX_METHOD_FLUX_FORCE_HPP

#include "flux_lax_friedrichs.hpp"
#include "flux_richtmyer.hpp"

namespace fluid {
namespace flux  {

/// The Force struct defines a callable object which solves for the flux
/// between two conservative states. The implementation is only defined for
/// conservative states so that no conversion is performed in the solver, since
/// it can be performed before invoking the method. The FORCE flux is defined
/// as: 
///   \begin{equation}
///     F_{i+1/2}^{force}(U_L, U_R)
///       = \frac{1}{2}
///           \left[ 
///             F_{i+1/2}^{LF}(U_L, U_R) + F_{i+1/2}^{RI}
///           \right]
///   \end{equation}
/// where $F_{i+1/2}^{LF}$ is the Lax-Friedrichs flux, and $F_{i+1/2}^{RI}$ is
/// the Richtmyer flux. For more information, see Toro, page 512.
struct Force {
 private:
  /// The Impl struct provides the implementation of the LF flux method for a
  /// given material type in a specific dimension.
  /// \tparam Mat   The type of the material used for the flux computation.
  /// \tparam Dim   The type of the dimension.
  template <typename Mat, typename Dim>
  struct Impl {
   private:
    /// Defines the type of material used for the flux computation.
    using material_t = std::decay_t<Mat>;
    /// Defines the data type used in the implementation.
    using value_t    = typename material_t::value_t;

    /// Defines the dimension to solve for.
    static constexpr auto dim = Dim();

    /// Stores a reference to the material to use for the implementation.
    material_t _mat;   //!< The material to solve for.
    value_t    _dtdh;  //!< Time delta for the iteration.
  
   public:
    /// Constructor which sets the material and scaling factor.
    /// \param[in] mat  The material for the computation.
    /// \param[in] dtdh The scaling factor for the computation.
    fluidity_host_device Impl(material_t mat, value_t dtdh)
    : _mat(mat), _dtdh(dtdh) {}

    /// Overload of operator() to compute the flux between the \p left and 
    /// \p right conservative states.
    /// \param[in] left   The left state for the Riemann problem.
    /// \param[in] right  The right state for the Riemann problem.
    /// \tparam    State  The type of the states.
    template <typename StateType>
    fluidity_host_device auto
    operator()(const StateType& ul, const StateType& ur) const
    {
      constexpr auto half = value_t{0.5};

      // Get invokable flux method functors
      const auto f_lf = LaxFriedrichs::get(_mat, _dtdh, dim);
      const auto f_ri = Richtmyer::get(_mat, _dtdh, dim);

      return half * (f_lf(ul, ur) + f_ri(ul, ur));
    }
  };

 public:
  /// Gets an instance of the flux evaluator for a give material \p mat and a
  /// given dimension.
  /// \param[in] mat    The material to use for the flux method.
  /// \param[in] dtdh   The space and time discretization factor.
  /// \tparam    Mat    The type of the material.
  /// \tparam    Dim    The type of the dimension.
  template <typename Mat, typename T, typename Dim>
  fluidity_host_device static auto get(const Mat& mat, T dtdh, Dim)
  {
    using flux_impl_t = Impl<Mat, Dim>;
    return flux_impl_t{mat, dtdh};
  }
};

}} // namespace fluid::flux

#endif // FLUIDITY_FLUX_METHOD_FLUX_FORCE_HPP