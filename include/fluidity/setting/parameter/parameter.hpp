//==--- fluidity/setting/parameter/parameter.hpp ----------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  parameter.hpp
/// \brief This file defines a static base class interface for a parameter of a
///        simulation whic can be configured from a file.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_PARAMETER_HPP
#define FLUIDITY_SETTING_PARAMETER_PARAMETER_HPP

#include <fluidity/setting/setting.hpp>
#include <fluidity/utility/string.hpp>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <fstream>

namespace fluid   {
namespace setting {

/// The PositionInfo class stores information about the position for a specific
/// dimension.
struct PositionInfo {
  std::string dim = "";   //!< The dimension for the position information.
  double      pos = 0.0;  //!< The location in the dimension.
};

/// The Parameter class defines a static interface for a parameter of a
/// simulation. 
/// \tparam ParamImpl The implementation of the parameter interface.
template <typename ParamImpl>
struct Parameter {
 private:
  /// Defines the implementation type of the paramter interface.
  using impl_t = ParamImpl;

 public:
  /// Checks if the \p seq contains the name for the parameter. This returns
  /// true if the name of the parameter is found in the \p seq, and false
  /// otherwise.
  /// \param[in] seq The sequence to search for the paramter name.
  bool contains_param(const std::string& seq) const
  {
    return seq.find(impl_t::name) != std::string::npos;
  }

  /// Returns the type of the identifier. This is the name of the identidier
  /// which is used in the settings file to set properties of the paramter with
  /// the type returned here.
  std::string type() const
  {
    return impl_t::name;
  }

  /// Returns a string which defines the required format of the parameter. That
  /// is, the identifiers and the types of their values. For example, for an
  /// "ambient_state" parameter with required properties "rho", "p", and "v_x",
  /// and optional properties "v_y", the required format should be:
  /// \begin{code}
  /// ambient_state : {
  ///   rho   : <value>,
  ///   p     : <value>,
  ///   v_x   : <value>[,]
  ///   [v_y  : <value>]   
  /// }
  /// \end{code}
  std::string required_format() const
  {
    return impl_t()->format();
  }

  /// Returns a vector of identifiers (names) of properties of the parameter
  /// which can be set. For example, if the parameter type is "ambient state",
  /// and the possible properties which can be set are "rho, p, v_x", then this
  /// should return a vector of strings "rho", "p", "v_x".
  std::vector<std::string> param_properties() const
  {
    return impl()->get_property_names();
  }

  /// Returns a string such that the configuration (properties and their values)
  /// of the parameter can be displayed. For example, for an "ambient_state"
  /// parameter with possible properties "rho", "p", and "v_x", the string
  /// should be:
  /// \begin{code}
  /// name : ambient_state
  ///   rho  : <value>
  ///   p    : <value>
  ///   v_x  : <value>
  /// \end{code}
  std::string printable_configuration() const
  {
    return impl()->printable_string();
  }

  /// Tries to set the value of the parameter using the \p setting. This returns
  /// true if the setting was succesfully set, otherwise it returs false.
  /// \param[in] setting The setting with the information to set the param with.
  bool try_set(const Setting& setting)
  {
    return impl()->try_set_from_setting(setting);
  }

  /// Checks if the \p property is one of the properties supported by the
  /// parameter or if it's a property of a property, etc, and returns true if
  /// the \p property is found, otherwise returns false.
  /// \param[in] property The name of the property to seach for.
  bool valid_property(const std::string& property) const
  {
    for (const auto& prop : param_properties())
    {
      if (prop.find(property) != std::string::npos)
        return true;
    }
    return false;
  }

  /// Returns true if the parameter is set.
  auto is_set() const -> bool
  {
    return _set;
  }
 
 protected:
  bool _set = false;  //!< If the paramter is set.

 private:
  /// Returns a pointer to a non-const implementation.
  impl_t* impl()
  {
    return static_cast<impl_t*>(this);
  }

  /// Returns a pointer to a const implementation.
  const impl_t* impl() const
  {
    return static_cast<const impl_t*>(this);
  }
};

/// Returns true if the type is a parameter.
/// \tparam T The type to check if is a paramter.
template <typename T>
static constexpr auto is_parameter_v = std::is_base_of<Parameter<T>, T>::value;

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_PARAMETER_HPP