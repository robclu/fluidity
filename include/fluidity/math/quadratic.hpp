//==--- fluidity/math/quadratic.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  quadratic.hpp
/// \brief This file defines the implementation of a quadratic class.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATH_QUADRATIC_HPP
#define FLUIDITY_MATH_QUADRATIC_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace math  {

/// The Quadratic class defines a class which stores the coefficients $a,b,c$
/// for a quadratic of the form $ax^2 + bx + c$, and which can solve the
/// equation as: 
///
/// \begin{equation}
///   x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
/// \end{equation}
///
/// This implementatio does not handle the negative discriminant case.
///
/// \tparam T The type of the data for the quadratic.
template <typename T>
struct Quadratic {
  /// Defines the type of the data for the quadratic.
  using value_t = std::decay_t<T>;
  /// Defines the type of this quadratic.
  using self_t  = Quadratic<T>;

  /// Defines a class for the roots of the quadratic equation solution.
  struct Roots {
    value_t smaller;  //!< The first (smaller) root of the solution.
    value_t larger;   //!< The second (larger) root of the solution.
  };

  value_t a = 0;  //!< Coefficient for the 2nd order term.
  value_t b = 0;  //!< Coefficient for the 1st order term.
  value_t c = 0;  //!< Coefficient for the constant term.

  /// Solves the quadratic equation. If the descriminant is negative, NaN is set
  /// for the resulting root.
  fluidity_host_device Roots solve() const
  {
    const auto denom = value_t{2} * a;
    const auto desc  = std::sqrt(b*b - value_t{4} * a * c);
    return Roots{(-b - desc) / denom, (-b + desc) / denom};
  }

  //==--- [Operator overloads] ---------------------------------------------==//
  
  /// Overload of the addition operator to add two quadratics together.
  /// \param[in] other The other quadratic to add with this one.
  /// \tparam    U     The data type for the other quadratic.
  template <typename U>
  fluidity_host_device self_t& operator+=(const Quadratic<U>& other)
  {
    a += other.a; b += other.b; c += other.b;
    return *this;
  }

  /// Overload of the subtraction operator to subtract one quadratic from this
  /// quadratic.
  /// \param[in] other The other quadratic to subtract from this one.
  /// \tparam    U     The data type for the other quadratic.
  template <typename U>
  fluidity_host_device self_t& operator-=(const Quadratic<U>& other)
  {
    a -= other.a; b -= other.b; c -= other.b;
    return *this;
  }
};

//==--- [Free functions] ---------------------------------------------------==//

/// Overload of the addition operator to add two quadratics together and return
/// a new quadratic with a data type of the left quadratic.
/// \param[in]  left  The left quadratic for the addition.
/// \param[in]  right The right quadratic for the addition.
/// \tparam     L     The data type for the left quadratic.
/// \tparam     R     The data type for the right quadratic.
template <typename L, typename R>
fluidity_host_device auto
operator+(const Quadratic<L>& l, const Quadratic<R>& r)
{
  return Quadratic<L>{l.a + r.a, l.b + r.b, l.c + r.c};
}

/// Overload of the subtraction operator to subtract one quadratic from another
/// and returna new quadratic with a data type of the left quadratic.
/// \param[in]  left  The left quadratic for the subtraction.
/// \param[in]  right The right quadratic for the subtraction.
/// \tparam     L     The data type for the left quadratic.
/// \tparam     R     The data type for the right quadratic.
template <typename L, typename R>
fluidity_host_device auto
operator-(const Quadratic<L>& l, const Quadratic<R>& r)
{
  return Quadratic<L>{l.a - r.a, l.b - r.b, l.c - r.c};
}

}} // namespace fluid::math


#endif // FLUIDITY_MATH_QUADRATIC_HPP
