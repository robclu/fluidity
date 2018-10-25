//==--- fluidity/setting/parameter/ambient_parameter.hpp --- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  ambient_parameter.hpp
/// \brief This file defines a class which implements the Parameter interface
///        to allow the ambient state of the simulation to be defined.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_AMBIENT_PARAMETER_HPP
#define FLUIDITY_SETTING_PARAMETER_AMBIENT_PARAMETER_HPP

#include "parameter.hpp"
#include <fluidity/setting/settings.hpp>
#include <fluidity/utility/string.hpp>
#include <tuple>

namespace fluid   {
namespace setting {

/// The AmbientParameter class allows the ambient state of the simulation to be
/// defined. This class is also used to determine the type of the state to use
/// for the simulation.
struct AmbientParameter : public Parameter<AmbientParameter> {
 private:
  /// Defines the format for a domain parameter.
  static constexpr const char* format_string = 
    "ambient_state : {\n"
    "  rho   : value,\n"
    "  p     : value,\n"
    "  v_x   : value[,]\n"
    "  [v_y  : value][,]\n"
    "  [v_z  : value]\n"
    "}\n";

 public:
  /// Defines the type of the container for the dimension information.
  using container_t = std::vector<Setting>;

  /// Defines the name of the parameter.
  static constexpr const char* name = "ambient_state";

  /// Returns a string which defines the required format for the parameter.
  std::string format() const
  {
    return format_string;
  }

  /// Returns a vector of identifiers for the names of settings for the domain.
  std::vector<std::string> get_property_names() const
  {
    return std::vector<std::string>{
      "rho", "p", "v_x", "v_y", "v_z"
    };
  }

  /// Returns a string with the information for the parameter.
  std::string printable_string() const
  {
    std::ostringstream stream;
    util::format_name_value(stream, "name", name, 0, 4);
    for (const auto& prop : _properties)
    {
      util::format_name_value(stream, prop.name, prop.value, 2, 13);
    }
    return stream.str();
  }

  /// Sets the values of the state from the \p setting.
  /// \param[in] setting The setting whose value contains the information
  ///                    to use to set the ambient parameter properties.
  bool try_set_from_setting(const Setting& setting)
  {
    if (!this->contains_param(setting.name))
      return false;

    auto props = Settings::from_string(setting.value);
    for (auto& prop : props)
    {
      if (!this->valid_property(prop.name))
      {
        std::cout << "Invalid property : " << prop.name << " for parameter : "
                  << name  << " can't set this property!\n";
        return false;
      }
      util::remove(prop.value, ',', '}', ' ', '\n');
      _properties.push_back(prop);
    }
    return true;
  }

 private:
  container_t _properties; //!< Properties for the ambient state.
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_AMBIENT_PARAMETER_HPP