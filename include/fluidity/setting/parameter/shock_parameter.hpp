//==--- fluidity/setting/parameter/shock_parameter.hpp ----- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  shock_parameter.hpp
/// \brief This file implements the Parameter interface to allow a shock to be
///        defined for a simulation.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_SHOCK_PARAMETER_HPP
#define FLUIDITY_SETTING_PARAMETER_SHOCK_PARAMETER_HPP

#include "parameter.hpp"
#include <fluidity/setting/settings.hpp>
#include <fluidity/utility/string.hpp>

namespace fluid   {
namespace setting {

/// The ShockParameter class specifies the format for defining a shock for a
/// simulation.
struct ShockParameter : public Parameter<ShockParameter> {
 private:
  /// Defines the format for a domain parameter.
  static constexpr const char* format_string = 
    "shock : {\n"
    "  mach : value,\n"
    "  x    : value,\n"
    "  [y   : value][,]\n"
    "  [z   : value][,]\n"
    "  [normal : { \n"
    "     x : value,\n"
    "     y : value,\n"
    "     z : value\n"
    "  }]\n"
    "}\n";

 public:
  /// Defines a list of the possible dimension values.
  static constexpr std::array<char, 3> pos_values = { 'x', 'y', 'z' };
  /// Defines the name of the parameter.
  static constexpr const char* name   = "shock";
  /// Defines the identifier for the mach number.
  static constexpr const char* mach   = "mach";
  /// Defines the identifier for the mach number.
  static constexpr const char* normal = "normal";

  /// Defines the type of the container for the dimension information.
  using container_t = std::vector<PositionInfo>;

  /// Returns a string which defines the required format for the parameter.
  std::string format() const
  {
    return format_string;
  }

  /// Returns a vector of identifiers for the names of settings for the domain.
  std::vector<std::string> get_property_names() const
  {
    return std::vector<std::string>{mach, "x", "y", "z", normal};
  }

  /// Returns a string with the information for the parameter.
  std::string printable_string() const
  {
    std::ostringstream stream;
    util::format_name_value(stream, "name", name, 0, 4);
    util::format_name_value(stream, mach, std::to_string(_mach), 2);
    for (const auto& pos : _positions)
    {
      util::format_name_value(stream, pos.dim, pos.pos, 2);
    }
    return stream.str();
  }

  /// Sets the information for a shock.
  /// \param[in] setting The setting whose value contains the information
  ///                    to use to set the shock information.
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
                  << name << ". Can't set this property.\n";
        return false;
      }
      if (std::strcmp(prop.name.c_str(), mach) == 0)
      {
        _mach = std::stod(prop.value);
        continue;
      }
      for (const auto pos : pos_values)
      {
        if (prop.name.find(pos) == std::string::npos)
        {
          continue;
        }

        if (prop.complex)
        {
          std::cout << "Failed to parse position component : " << pos
                    << " for shock. Shock parameter must have the following "
                    << "format:\n" << format_string;
          return false;
        }
        _positions.emplace_back();
        auto &p = _positions.back();
        p.dim = prop.name; p.pos = std::stod(prop.value);
        util::remove(p.dim, ' ');
        break;
      }
    }
    if (_mach <= 0.0 || _positions.size() == 0)
    {
      std::cout << "Failed to parse " << name << " correctly. "
                << "Shock must have the following format:\n" 
                << format_string;
      return false;
    }
    this->_set = true;
    return this->_set;
  }

 private:
  double      _mach = 0.0;  //!< Mach number for the shock.
  container_t _positions;   //!< Positions of the shock.
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_SHOCK_PARAMETER_HPP