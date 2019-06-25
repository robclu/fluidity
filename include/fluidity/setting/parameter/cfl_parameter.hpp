//==--- fluidity/setting/parameter/cfl_parameter.hpp ------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  cfl_parameter.hpp
/// \brief This file defines a class which implements the Parameter interface
///        to allow the CFL number for the simulation to be set.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMTER_CFL_PARAMETER_HPP
#define FLUIDITY_SETTING_PARAMTER_CFL_PARAMETER_HPP

#include "parameter.hpp"
#include <fluidity/setting/settings.hpp>
#include <fluidity/utility/string.hpp>

namespace fluid   {
namespace setting {

/// The CflParameter class allows the cfl number to be set for the simulation.
struct CflParameter : public Parameter<CflParameter> {
 private:
  /// Defines the format for a domain parameter.
  static constexpr const char* format_string = "clf : value\n";
  /// Defines the min value of the CFL.
  static constexpr double cfl_min            = 0.0;
  /// Defines the max value of the CFL.
  static constexpr double cfl_max            = 1.0;

 public:
  /// Defines the name of the parameter.
  static constexpr const char* name = "cfl";

  /// Returns a string which defines the required format for the parameter.
  std::string format() const
  {
    return format_string;
  }

  /// Returns a vector of identifiers for the names of settings for the domain.
  std::vector<std::string> get_property_names() const
  {
    return std::vector<std::string>{"cfl"};
  }

  /// Returns a string with the information for the parameter.
  std::string printable_string() const
  {
    std::ostringstream stream;
    util::format_name_value(stream, "name", name, 0, 4);
    util::format_name_value(stream, name, std::to_string(_cfl), 2);
    return stream.str();
  }

  /// Sets the value of the cfl parameter from the setting.
  /// \param[in] setting The setting whose value contains the information
  ///                    to use to set the cfl parameter.
  bool try_set_from_setting(const Setting& setting)
  {
    if (!this->contains_param(setting.name))
      return false;

    if (setting.complex)
    {
      std::cout << "Parameter : " << name << " cannot be complex.\n"
                << "Format for "  << name << " is:\n" << format_string;  
      return false;
    }
    _cfl = std::stod(setting.value);
    this->_set = _cfl > cfl_min &&_cfl <= cfl_max;
    return this->_set;
  }

  /// Returns the value of the CFL.
  double cfl() const { return _cfl; }

 private:
  double _cfl = 0.0;  //!< The value of the cfl.
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_AMBIENT_PARAMETER_HPP