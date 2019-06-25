//==--- fluidity/limiting/limiter_traits.hpp- ----------------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  limiter_traits.hpp
/// \brief This file defines type traits of limiters.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LIMITING_LIMITER_TRAITS_HPP
#define FLUIDITY_LIMITING_LIMITER_TRAITS_HPP

#include <fluidity/state/state_traits.hpp>

namespace fluid {
namespace limit {

/// The LimiterTraits class defines traits of limiters.
/// \tparam Limiter The limiter to get the traits for.
template <typename Limiter>
class LimiterTraits;

/// The LimitForm stuct defines a type which defines the form for which the
/// states are limited on.
/// \tparam Form The form of the variables to limit on.
template <state::FormType Form>
struct LimitForm {
  /// Defines the form to limt on.
  static constexpr auto form = Form;
};

/// Defines an alias for limiting on primitive variables.
using prim_form_t = LimitForm<state::FormType::primitive>;

/// Defines an alias to limit on conservative variables.
using cons_form_t = LimitForm<state::FormType::conservative>;

//== Forward declarations and specializations ------------------------------==//

/// Forward declaration of the Void limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form> struct Void;

/// Specialization for limiter traits for the Void limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form>
struct LimiterTraits<Void<Form>>
{
  /// Defines the type of the limiter.
  using limiter_t = Void<Form>;
  /// Defines the form used for the limiting.
  using form_t    = Form;
  
  /// Defines the form of the state to limit on.
  static constexpr auto form = form_t::form;

  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = std::size_t{0};
};

/// Forward declaration of the Linear limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form> struct Linear;

/// Specialization for limiter traits for the VanLeer limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form>
struct LimiterTraits<Linear<Form>>
{
  /// Defines the type of the limiter.
  using limiter_t = Linear<Form>;
  /// Defines the form used for the limiting.
  using form_t    = Form;
  
  /// Defines the form of the state to limit on.
  static constexpr auto form = form_t::form;

  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = std::size_t{1};
};

/// Forward declaration of the VanLeer limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form> struct VanLeer;

/// Specialization for limiter traits for the VanLeer limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form>
struct LimiterTraits<VanLeer<Form>>
{
  /// Defines the type of the limiter.
  using limiter_t = VanLeer<Form>;
  /// Defines the form used for the limiting.
  using form_t    = Form;
  
  /// Defines the form of the state to limit on.
  static constexpr auto form = form_t::form;

  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = std::size_t{4};
};

/// Forward declaration of the VanLeer limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form> struct VanLeerUnsplit;

/// Specialization for limiter traits for the VanLeer limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form>
struct LimiterTraits<VanLeerUnsplit<Form>>
{
  /// Defines the type of the limiter.
  using limiter_t = VanLeerUnsplit<Form>;
  /// Defines the form used for the limiting.
  using form_t    = prim_form_t;
  
  /// Defines the form of the state to limit on.
  static constexpr auto form = form_t::form;

  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = std::size_t{4};
};

/// Forward declaration of the Superbee limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form> struct Superbee;

/// Specialization for limiter traits for the Superbee limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form>
struct LimiterTraits<Superbee<Form>>
{
  /// Defines the type of the limiter.
  using limiter_t = Superbee<Form>;
   /// Defines the form used for the limiting.
  using form_t    = Form; 

  /// Defines the form of the state to limit on.
  static constexpr auto form = form_t::form;
  
  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = std::size_t{4};
};


/// Forward declaration of the Superbee unsplit limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form> struct SuperbeeUnsplit;

/// Specialization for limiter traits for the Superbee unsplit limiter.
/// \tparam Form The form of the state to limit on.
template <typename Form>
struct LimiterTraits<SuperbeeUnsplit<Form>>
{
  /// Defines the type of the limiter.
  using limiter_t = SuperbeeUnsplit<Form>;
   /// Defines the form used for the limiting.
  using form_t    = prim_form_t; 

  /// Defines the form of the state to limit on.
  static constexpr auto form = form_t::form;
  
  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = std::size_t{4};
};

}} // namespace fluid::limit

#endif // FLUIDITY_LIMITING_LIMITER_TRAITS_HPP