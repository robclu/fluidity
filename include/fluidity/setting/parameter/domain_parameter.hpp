//==--- fluidity/setting/parameter/domain_parameter.hpp ---- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  domain_parameter.hpp
/// \brief This file implements the Parameter interface to allow the domain of a
///        simulation to be 
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_PARAMETER_DOMAIN_PARAMETER_HPP
#define FLUIDITY_SETTING_PARAMETER_DOMAIN_PARAMETER_HPP

#include "parameter.hpp"
#include <fluidity/setting/settings.hpp>
#include <array>
#include <cstring>

namespace fluid   {
namespace setting {

/// Defines a struct to represent the information for a dimension of a domain.
struct DomainDimInfo {
  char    dim;    //!< The dimension for the information.
  double  start;  //!< The start value of the dimension.
  double  end;    //!< The end value of the dimension.
};

/// The DomainParameter class allows parameters of the domain to be specified.
struct DomainParameter : public Parameter<DomainParameter> {
 private:
  /// Defines the format for a domain parameter.
  static constexpr const char* format_string = 
    "domain : {\n"
    "  resolution : value,\n"
    "  x          : { start : value, end : value }[,]\n"
    "  [y         : { start : value, end : value }][,]\n"
    "  [z         : { start : value, end : value }]\n"
    "}\n";

 public:
  /// Defines a list of the possible dimension values.
  static constexpr std::array<char, 3> dim_values = { 'x', 'y', 'z' };
  /// Defines the identifier for the resolution.
  static constexpr const char* resolution = "resolution";
  /// Defines the name of the paramter.
  static constexpr const char* name       = "domain";

  /// Defines the type of the container for the dimension information.
  using container_t = std::vector<DomainDimInfo>;

  /// Returns a string which defines the required format for the parameter.
  std::string format() const
  {
    return format_string;
  }

  /// Returns a vector of identifiers for the names of settings for the domain.
  std::vector<std::string> get_property_names() const
  {
    return std::vector<std::string>{"x", "y", "z", resolution};
  }

  /// Returns the number of dimensions for which the domain parameter has
  /// information.
  std::size_t domain_dimensions() const
  {
    return _dim_info.size();
  }

  /// Returns a string with the information for the parameter.
  std::string printable_string() const
  {
    std::ostringstream stream;
    util::format_name_value(stream, "name", name, 0, 4);
    util::format_name_value(stream, resolution, std::to_string(_resolution), 2);
    for (const auto& dim : _dim_info)
    {
      std::string s = "[ ";
      s += std::to_string(dim.start) + ", " + std::to_string(dim.end) + " ]";
      util::format_name_value(stream, dim.dim, s, 2);
    }
    return stream.str();
  }

  /// Sets the dimension information for a dimension.
  /// \param[in] setting The setting whose value contains the information
  ///                    to use to set the domain.
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
                  << name << ". Can't set this property!\n";
        return false;
      }
      if (std::strcmp(prop.name.c_str(), resolution) == 0)
      {
        _resolution = std::stod(prop.value);
        continue;
      }
      for (const auto dim : dim_values)
      {
        if (prop.name.find(dim) == std::string::npos)
        {
          continue;
        }

        if (!setting.complex)
        {
          std::cout << "Failed to parse dimension component : " << dim
                    << "for domain. Domain parameter must have the following "
                    << "format:\n" << format_string;
          return false;
        }
        auto dim_settings = Settings::from_string(prop.value);
        
        auto info  = DomainDimInfo();
        info.dim   = dim;
        info.start = stod(dim_settings.front().value);
        info.end   = stod(dim_settings.back().value);
        std::cout << info.start << " ::: " << info.end << "\n";
        _dim_info.push_back(std::move(info));
        break;
      }
    }
    if (_resolution <= 0.0 || _dim_info.size() == 0)
    {
      std::cout << "Failed to parse " << name << " correctly. "
                << "The following format is required:\n" << format_string;
      return false;
    }
    return true;
  }

 private:
  double      _resolution = 0.0;  //!< Resolution for the domain.
  container_t _dim_info;          //!< Information for the dimensions.
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_PARAMETER_DOMAIN_PARAMETER_HPP