//==--- fluidity/flux_method/flux_lax_friedrichs.hpp ------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_lax_friedrichs.hpp
/// \brief This file defines an implementation of the computation of the flux
///        between two states using the Lax-Friedrichs method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_FLUX_METHOD_FLUX_LAX_FRIEDRICHS_HPP
#define FLUIDITY_FLUX_METHOD_FLUX_LAX_FRIEDRICHS_HPP

#include <fluidity/state/state_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace flux  {

/// The LaxFriedrichs struct defines a callable object which solves for the flux
/// between two conservative states. The implementation is only defined for
/// conservative states so that no conversion is performed in the solver, since
/// it can be performed before invoking the method. The Lax-Friedrichs flux is
/// defined as: 
///   \begin{equation}
///     F_{i+1/2}^{LF}(U_L,U_R) = 
///       \frac{1}/{2} \left[ F(U_L) + F(U_R) \right] +
///       \frac{1}{2} \frac{\delta x}{\delta t} \left[ U_L - U_R \right]
///   \end{equation}
/// For more information, see Toro, page 512.
struct LaxFriedrichs {
 private:
  /// The Impl struct provides the implementation of the LF flux method for a
  /// given material type in a specific dimension.
  /// \tparam Material The type of the material used for the flux computation.
  /// \tparam Value    The value which defines the dimension.
  template <typename Material, std::size_t Value>
  struct Impl {
   private:
    /// Defines the type of material used for the flux computation.
    using material_t = std::decay_t<Material>;
    /// Defines the data type used in the implementation.
    using value_t    = typename material_t::value_t;

    /// Stores a reference to the material to use for the implementation.
    material_t _mat;   //!< The material to solve for.
    value_t    _dhdt;  //!< Time delta for the iteration.

   public:
    /// Constructor which sets the material and scaling factor.
    /// \param[in] mat  The material for the computation.
    /// \param[in] dtdh The scaling factor for the computation.
    fluidity_host_device Impl(material_t mat, value_t dtdh)
    : _mat(mat), _dhdt(value_t{1.0} / dtdh) {}

    /// Overload of operator() to compute the flux between the \p left and 
    /// \p right conservative states.
    /// \param[in] left   The left state for the Riemann problem.
    /// \param[in] right  The right state for the Riemann problem.
    /// \tparam    State  The type of the states.
    template <typename State>
    fluidity_host_device auto operator()(const State& ul, const State& ur) const
    {
      constexpr auto dim  = Dimension<Value>();
      constexpr auto half = value_t{0.5};
      return half * (ul.flux(_mat, dim) + 
                     ur.flux(_mat, dim) + 
                     _dhdt * (ul - ur)  );
    }
  };

 public:
  /// Gets an instance of the flux evaluator for a give material \p mat and a
  /// given dimension.
  /// \param[in] mat      The material to use for the flux method.
  /// \param[in] dtdh     The space and time discretization factor.
  /// \tparam    Material The type of the material.
  /// \tparam    Value    The value which defines the dimension. 
  template <typename Material, typename T, std::size_t Value>
  fluidity_host_device static auto 
  get(const Material& mat, T dtdh, Dimension<Value> /*dim*/)
  {
    using flux_impl_t = Impl<Material, Value>;
    return flux_impl_t{mat, dtdh};
  }
};

}} // namespace fluid::flux

#endif // FLUIDITY_FLUX_METHOD_FLUX_LAX_FRIEDRICHS_HPP
