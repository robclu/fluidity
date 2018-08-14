//==--- fluidity/setting/option_tuple.hpp ------------------ -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  option_tuple.hpp
/// \brief This file defines a class to hold a tuple of option types, where each
///        option is a type.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_OPTION_TUPLE_HPP
#define FLUIDITY_SETTING_OPTION_TUPLE_HPP

#include "option_builder.hpp"
#include <utility>

namespace fluid   {
namespace setting {
namespace detail  {

/// The OptionElement stuct stores the index of the option in the option tuple,
/// as well as the type of an option and if the option is the final option.
/// \tparam Index The index of the element.
/// \tparam Opt   The type of the option which the element holds.
/// \tparam Last  If the element is the last (final option to be applied)
///               element in the option list.
template <std::size_t Index, typename Opt, bool Last>
struct OptionElement;

} /// namespace detail

/// Gets the option element \p o at position I in an option list. Because the
/// option list is implemented using inheritance, this overload will only be
/// enabled for the correct option element. This function is also only provided
/// for reference type elements.
/// \param[in] op_element   The element to get.
/// \tparam    I            The index of the element to get.
/// \tparam    Opt          The type of the option the element holds.
/// \tparam    Last         If the element is the last element.
template <std::size_t I, typename Opt, bool Last>
auto& opt_get(detail::OptionElement<I, Opt, Last>& op_element)
{
  return op_element;
}

/// Gets the option element \p o at position I in an option list. Because the
/// option list is implemented using inheritance, this overload will only be
/// enabled for the correct option element. This function is also only provided
/// for reference type elements.
/// \param[in] op_element   The element to get.
/// \tparam    I            The index of the element to get.
/// \tparam    Opt          The type of the option the element holds.
/// \tparam    Last         If the element is the last element.
template <std::size_t I, typename Opt, bool Last>
const auto& opt_get(const detail::OptionElement<I, Opt, Last>& op_element)
{
  return op_element;
}

namespace detail {

/// The OptionElement stuct stores the index of the option in the option tuple,
/// as well as the type of an option and if the option is the final option. This
/// specialization is for elements which are not the last elements, and the
/// append method adds the type of the held option to a list of option choices.
/// \tparam Index The index of the element.
/// \tparam Opt   The type of the option which the element holds.
template <std::size_t Index, typename Opt>
struct OptionElement<Index, Opt, false>
{
  /// Defines the type of the option.
  using option_type_t = Opt;

  /// Defines the index of this option.
  static constexpr auto index      = Index;
  /// Defines the index of the next option to append to the list of choices.
  static constexpr auto next_index = index - 1;

  /// Appends the option type to the list of Choices and defines the next option
  /// element to use to get the next option type to append to the list.
  /// \param[in] options A reference to the list of all options.
  /// \param[in] base    A reference to the base type to modify.
  /// \tparam    Choices A list of option choices.
  /// \tparam    Derived The type of the derived class to eventually build.
  /// \tparam    OpList  A list of all options.
  /// \tparam    Base    The type of the base class to eventually build.
  template <typename Choices, typename Derived, typename OpList, typename Base>
  void append(const OpList& options, Base& base) const
  {
    const auto& next_opt = opt_get<next_index>(options);
    option.template append<Choices, Derived>(options, next_opt, base);
  }

  option_type_t option; //!< The option which the element holds.
};


/// The OptionElement stuct stores the index of the option in the option tuple,
/// as well as the type of an option and if the option is the final option. This
/// specialization is for elements which are the last element in the option list
/// and the append method finishes the creation of the choice list.
/// \tparam Index The index of the element.
/// \tparam Opt   The type of the option which the element holds.
template <std::size_t Index, typename Opt>
struct OptionElement<Index, Opt, true>
{
  /// Defines the type of the option.
  using option_type_t = Opt;

  /// Defines the index of this option.
  static constexpr auto index = Index;

  /// Finishes the creation of the option list from the choices by calling the
  /// finish method on the option.
  /// \param[in] options A reference to the list of all options.
  /// \param[in] base    A reference to the base type to modify.
  /// \tparam    Choices A list of option choices.
  /// \tparam    Derived The type of the derived class to eventually build.
  /// \tparam    OpList  A list of all options.
  /// \tparam    Base    The type of the base class to eventually build.    
  template <typename Choices, typename Derived, typename OpList, typename Base>
  void append(const OpList& options, Base& base) const
  {
    option.template finish<Choices, Derived>(options, base);
  }
    
  option_type_t option; //!< The option which the element holds.
};

/// The OptionTuple struct implements a list of different option types,
/// where each option holds different types, which can be set at runtime.
/// \tparam Indices The indices of the option elements.
/// \tparam Options The types of the options.
template <typename Indices, typename... Options>
struct OptionTuple;

/// The OptionTuple struct implements a list of different option types,
/// where each option holds different types, which can be set at runtime. This
/// is a specialization for an index sequence which allows the option tuple to
/// inherit from each of the option elements.
/// \tparam N       The index of the option elements.
/// \tparam Options The types of the options.
template <std::size_t... N, typename... Options>
struct OptionTuple<std::index_sequence<N...>, Options...> :
OptionElement<N, Options, (N==0)>... {

 /// Defines the number of options in the tuple.
 static constexpr auto size           = sizeof...(Options);
 /// Defines the index of the first option to get.
 static constexpr auto first_option_v = size - 1;
 /// Creates a unique pointer to the Base type of the Derived<...> type, where
 /// the template parameters of the derived type are filled based on the choices
 /// from the Options.
 /// \tparam base    A reference to the base class to set.
 /// \tparam Derived The type of the derived class to create.
 /// \tparam Base    The type of the base class. 
 template <typename Derived, typename Base>
 void create(Base& base) const
 {
   const auto& first_option = opt_get<first_option_v>(*this);
   first_option.template append<empty_builder_t, Derived>(*this, base);
 }
};

} // namespace detail

/// The OptionTuple struct implements a list of different option types,
/// where each option holds different types (defined at compile time) but 
/// which can be set at runtime.
/// \tparam Indices The indices of the option elements.
/// \tparam Options The types of the options.
template <typename... Options>
struct OptionTuple final :
detail::OptionTuple<std::make_index_sequence<sizeof...(Options)>, Options...> {
  /// Defines the type of the index sequence.
  using indices_t = std::make_index_sequence<sizeof...(Options)>;
  /// Defines the type of the base imlpementation.
  using impl_t    = detail::OptionTuple<indices_t, Options...>;

 /// Creates a unique pointer to the Base type of the Derived<...> type, where
 /// the template parameters of the derived type are filled based on the choices
 /// from the Options.
 /// \tparam base    A reference to the base class to set.
 /// \tparam Derived The type of the derived class to create.
 /// \tparam Base    The type of the base class. 
 template <typename Derived, typename Base>
 void create(Base& base) const
 {
   impl_t::template create<Derived>(base);
 }
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_OPTION_TUPLE_HPP