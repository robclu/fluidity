//==--- fluidity/setting/option.hpp ------------------------ -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  option.hpp
/// \brief This file defines a class which provides the functionality for
///        creating an option for a setting.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_OPTION_HPP
#define FLUIDITY_SETTING_OPTION_HPP

//#include <fluidity/algorithm/unrolled_for.hpp>
#include "option_tuple.hpp"
#include <array>
#include <cstring>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace fluid   {
namespace setting {

/// The Option class is a CRTP class which defines the interface for an option
/// for a setting.
/// \tparam OpImpl The implementation of the option which specifies the details
///                of the option.
template <typename OpImpl>
struct Option {
  /// Defines the type of the option implementation.
  using op_impl_t = OpImpl;
   
  /// Returns true if the \p op_type matches the type of the option
  /// implementation.
  /// \param[in] op_type The type to check the validity of.
  bool is_valid_option_type(std::string op_type) const
  {
    return std::strcmp(op_impl_t::type, op_type.c_str()) == 0;
  }

  /// Sets the value of the option to \p value if ]p value is a valid option,
  /// returining true of the value was set, and false otherwise.
  /// \param[in] value The value to set the option to.
  bool set_option_value(std::string value)
  {
    for (const auto choice : get_option_choice_list())
    {
      if (!(std::strcmp(choice.c_str(), value.c_str()) == 0)) { continue; }
                
      _value = value;
      return true;
    }
    return false;
  }

  /// Appends the appropriate option from the option implementation to the list
  /// of choices based on the choices in the option implementation and the value
  /// set ffor the option at runtime. At compile time, a switch is made for each
  /// of the option choices defined in the implementation, and the correct one
  /// is chosen at runtime.
  /// \param[in] options An option list of all the options to apply.
  /// \param[in] next    The next option to append to the choices.
  /// \param[in] base    The type of the base class object to create.
  /// \tparam    Choices A list of choice types to be appended to.
  /// \tparam    Derived The type of the derived class to create.
  /// \tparam    OpList  The type of the options to set.
  /// \tparam    NextOp  The type of the next option to append.
  /// \tparam    Base    The type of the base class to create.
  template < typename Choices
           , typename Derived
           , typename OpList
           , typename NextOp
           , typename Base>
  void append(const OpList& options, const NextOp& next, Base& base) const
  {
    bool match = false;
    for_each_option(op_impl_t::choice_list(), [&] (const auto& choice)
    {
      if (_value == choice.value)
      {
        log_set_message();
        match = true;
        using new_t         = typename std::decay_t<decltype(choice)>::type;
        using new_choices_t = typename Choices::template appended_t<new_t>;
        next.template append<new_choices_t, Derived>(options, base);
      }
    });
    if (!match)
    {
      log_set_message();
      const auto choice = get_default_choice();
      using new_t         = typename std::decay_t<decltype(choice)>::type;
      using new_choices_t = typename Choices::template appended_t<new_t>;
      next.template append<new_choices_t, Derived>(options, base);
    }
  }
    
  /// Finishes creating options by invoking the builder object to set the \p
  /// base class to a new pointer of the Derived type using the Choices as
  /// template paramters for the Derived type. This overload is only enables
  /// when the list of Choices (after applying this final choice) is the same
  /// length as the list of options to set.
  /// \param[in] options An option list of all the options to apply.
  /// \param[in] base    The type of the base class object to create.
  /// \tparam    Choices A list of choice types to be appended to.
  /// \tparam    Derived The type of the derived class to create.
  /// \tparam    OpList  The type of the options to set.
  /// \tparam    Base    The type of the base class to create.
  template < typename Choices
           , typename Derived
           , typename OpList
           , typename Base
           , std::enable_if_t<(Choices::size + 1 == OpList::size), int> = 0>
  void finish(const OpList& options, Base& base) const
  {
    bool match = false;
    for_each_option(op_impl_t::choice_list(), [&] (const auto& choice)
    {
      if (_value == choice.value)
      {
        log_set_message();
        match = true;
        using new_t         = typename std::decay_t<decltype(choice)>::type;
        using new_choices_t = typename Choices::template appended_t<new_t>;
        new_choices_t::template build<Derived>(base);
      }
    });
    if (!match)
    {
      log_set_message();
      const auto choice = get_default_choice();
      using new_t         = typename std::decay_t<decltype(choice)>::type;
      using new_choices_t = typename Choices::template appended_t<new_t>;
      new_choices_t::template build<Derived>(base);
    }
  }
    
  /// Finishes creating options by invoking the builder object to set the \p
  /// base class to a new pointer of the Derived type using the Choices as
  /// template paramters for the Derived type. This overload is enabled when the
  /// number of appended choices does not match the number of options to append
  /// to the choice list. It stops the iteration for all paths which will not
  /// result in a valid setting.
  /// \param[in] options An option list of all the options to apply.
  /// \param[in] base    The type of the base class object to create.
  /// \tparam    Choices A list of choice types to be appended to.
  /// \tparam    Derived The type of the derived class to create.
  /// \tparam    OpList  The type of the options to set.
  /// \tparam    Base    The type of the base class to create
  template < typename Choices
           , typename Derived
           , typename OpList
           , typename Base
           , std::enable_if_t<(Choices::size + 1 != OpList::size), int> = 0>
  void finish(const OpList& options, Base& base) const {}
    
 private:
  std::string _value; //!< The value of the option.
  
  /// Returns a pointer to a non-const implementation.
  op_impl_t* op_impl()
  {
    return static_cast<op_impl_t*>(this);
  }

  /// Returns a pointer to a const implementation.
  const op_impl_t* op_impl() const
  {
    return static_cast<const op_impl_t*>(this);
  }
    
  /// Returns an array of possible choices for the option.
  auto get_option_choice_list() const 
  {
    std::vector<std::string> choices;
    for_each_option(op_impl_t::choice_list(), [&] (const auto& choice)
    {
      choices.push_back(choice.value);
    });
    return choices;
  }
    
  /// Returns the default choice for the option.
  /// \param[in] invalid_choice The invalid choice which was given.
  auto get_default_choice() const
  {
    auto choice = opt_get<0>(op_impl_t::choice_list());
    std::cout << "\nChoice : '"                << _value 
              << "' is invalid for setting : " << op_impl_t::type 
              << "\nChoices for the "          << op_impl_t::type
              << " setting are:\n";
    for (const auto& c : get_option_choice_list())
    {
      std::cout << "\t- " << c << "\n";
    }
    std::cout << "Using the default choice:\n\t- " << choice.value << "\n";
    return choice;
  }

  /// Logs the type of the option and the value.
  void log_set_message() const
  {
    /* auto s = std::string("setting option type :")
              + op_impl_t::type
              + " with value : "
              + _value;
      logging::log(logging::logger_t, s);
    */
  }
};

/// Returns true if the template type T is derived from an Option.
/// \tparam T The type to check if is an option.
template <typename T>
constexpr auto is_option_v = std::is_base_of<Option<T>, T>::value;

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_OPTION_HPP