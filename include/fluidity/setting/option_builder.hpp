//==--- fluidity/setting/option_builder.hpp ---------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  option_builder.hpp
/// \brief This file defines a utility class which provides the functionality
///        building a derived class from an OptionTuple.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_OPTION_BUILDER_HPP
#define FLUIDITY_SETTING_OPTION_BUILDER_HPP

#include <cassert>
#include <memory>

namespace fluid   {
namespace setting {
namespace detail  {

/// The OptionBuilder struct is used to build a base class through a templated
/// derived class, where the template paramters for the derived class are
/// provided through the OptionBuilder. It also provided functionality to
/// extend the template parameter list for the derived class.
/// \tparam Ts The types to build the derived class with.
template <typename... Ts>
struct OptionBuilder {
  /// Defines the number of type parameters for the derived class.
  static constexpr auto size = sizeof...(Ts);
  
  /// Defines the type of the builder with an appended paramter.
  /// \tparam T The type to append.
  template <typename T> using appended_t = OptionBuilder<T, Ts...>;
  
  /// Builds a Derived<Ts...> unique_ptr to the Base, modifying the \p base
  /// reference. This checks that Derived<Ts...> is derived from Base, and if
  /// not, a compile time error is generated.
  /// \param[in] base     The base class unique_ptr to modify.
  /// \tparam    Derived  The type of the derived class.
  /// \tparam    Base     The type of the base class.
  template <typename Derived, typename Base>
  static auto build(Base& base)
  {
    using derived_type_t = typename Derived::template make_type_t<Ts...>;
/*
    static_assert(std::is_base_of<Base, derived_type_t>::value,
                  "Derived type is not derived from the base class, "
                  "so a pointer to the base cannot be created through "
                  "this derived type.");
*/
    base = std::move(std::make_unique<derived_type_t>());
  }
};

/// Specialization of the OptionBuilder class for the case that there are no
/// options. 
template <> struct OptionBuilder<> {
  /// Defines the number of type parameters for the derived class.  
  static constexpr auto size = std::size_t{0};
  
  /// Defines the type of the builder with an appended paramter.
  /// \tparam T The type to append.
  template <typename T> using appended_t = OptionBuilder<T>;

  /// This simply does nothing since there are no type parameters to instantiate
  /// the derived type with.
  /// \param[in] base     The base class unique_ptr to modify.
  /// \tparam    Derived  The type of the derived class.
  /// \tparam    Base     The type of the base class.
  //template <typename Derived, typename Base>
  //static auto build(Base& base) {}
};

} // namespace detail

/// Defines the type of an initial (empty) builder to start building with.
using empty_builder_t = detail::OptionBuilder<>;

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_OPTION_BUILDER_HPP
