//==--- fluidity/flux_method/flux_richtmyer.hpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_richtmyer.hpp
/// \brief This file defines an implementation of the computation of the flux
///        between two states using the Richtmyer method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_FLUX_METHOD_FLUX_RICHTMYER_HPP
#define FLUIDITY_FLUX_METHOD_FLUX_RICHTMYER_HPP

#include <fluidity/state/state_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace flux  {

/// The Richtmyer struct defines a callable object which solves for the flux
/// between two conservative states. The implementation is only defined for
/// conservative states so that no conversion is performed in the solver, since
/// it can be performed before invoking the method. The Richtmyer flux is
/// defined as: 
///   \begin{equation}
///     F_{i+1/2}^{RI}(U_{i+1/2}^{RI}
///   \end{equation}
/// where
///   \begin{equation}
///     U_{i+1/2}^{RI} 
///       = U_{i+1/2}^{RI}(U_L, U_R) 
///       = \frac{1}{2} (U_L + U_R) +
///           \frac{1}{2} \frac{\delta t}{\delta x } 
///           \left[ 
///             F(U_L) - F(U_R)
///           \right]
///   \end{equation}
/// For more information, see Toro, page 512.
struct Richtmyer {
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

    /// Defines the dimension for the implementation.
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
    /// \param[in] left       The left state for the Riemann problem.
    /// \param[in] right      The right state for the Riemann problem.
    /// \tparam    StateType  The type of the states.
    template <typename StateType>
    fluidity_host_device auto
    operator()(const StateType& ul, const StateType& ur) const
    {
      constexpr auto half = value_t{0.5};
      auto u_ri = StateType
      {
        half * (ul + ur + _dtdh * (ul.flux(_mat, dim) - ur.flux(_mat, dim)))
      };
      return u_ri.flux(_mat, dim);
    }
  };

 public:
  /// Gets an instance of the flux evaluator for a give material \p mat and a
  /// given dimension.
  /// \param[in] mat    The material to use for the flux method.
  /// \param[in] dtdh   The space and time discretization factor.
  /// \tparam    Mat    The type of the material.
  /// \tparam    T      The type of the scaling factor.
  /// \tparam    Dim    The type of the dimension.
  template <typename Mat, typename T, typename Dim>
  fluidity_host_device static auto get(const Mat& mat, T dtdh, Dim)
  {
    using flux_impl_t = Impl<Mat, Dim>;
    return flux_impl_t{mat, dtdh};
  }
};

}} // namespace fluid::flux

#endif // FLUIDITY_FLUX_METHOD_FLUX_RICHTMYER_HPP